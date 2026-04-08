package cpullmapi

import (
	"context"
	"fmt"
	"os"
	"reflect"
	"runtime"
	"sync"

	"golang.org/x/sys/unix"
)

type TaskFunc func()

type ExecutorPool []chan TaskFunc

func NewExecutorPool(num int) (ExecutorPool, error) {
	if num < 1 {
		return nil, fmt.Errorf("num must be greater than 0")
	}
	pool := make(ExecutorPool, num)
	for i := range num {
		pool[i] = make(chan TaskFunc)
	}
	return pool, nil
}

func (p ExecutorPool) Dispatch(task TaskFunc) {
	cases := make([]reflect.SelectCase, len(p))
	v := reflect.ValueOf(task)
	for i, ch := range p {
		cases[i] = reflect.SelectCase{
			Dir:  reflect.SelectSend,
			Chan: reflect.ValueOf(ch),
			Send: v,
		}
	}
	reflect.Select(cases)
}

func (p ExecutorPool) Start(threads int) error {
	numCPU := runtime.NumCPU()
	if threads < 1 || threads > numCPU {
		return fmt.Errorf("threads must be between 1 and %d", numCPU)
	}
	cpuIndexStep := numCPU / len(p)
	if cpuIndexStep == 0 {
		cpuIndexStep = 1
	}

	for i, ch := range p {
		go func(ch chan TaskFunc, startCPUIndex int) {
			runtime.LockOSThread()
			defer runtime.UnlockOSThread()

			// set affinity
			cpuSet := unix.CPUSet{}
			for j := range threads {
				// fmt.Printf("setting %d affinity to %d\n", unix.Gettid(), (startCPUIndex+cpuIndexStep*j)%numCPU)
				cpuSet.Set((startCPUIndex + cpuIndexStep*j) % numCPU)
			}
			err := unix.SchedSetaffinity(unix.Gettid(), &cpuSet)
			if err != nil {
				fmt.Fprintf(os.Stderr, "failed to set affinity for thread %d: %v", unix.Gettid(), err)
				return
			}

			for task := range ch {
				if task == nil {
					// end
					break
				}
				task()
			}
		}(ch, i)
	}

	return nil
}

func (p ExecutorPool) Stop() {
	for _, ch := range p {
		close(ch)
	}
}

type Closeable interface {
	Close() error
}

type rpStub[T Closeable] struct {
	name string
	mu   sync.Mutex

	resourceUsage int
	obj           T
}

type rpDesc[T Closeable] struct {
	resourceRequired int
	objFactory       func() (T, error)
}

type ResourcePool[T Closeable] struct {
	totalResource int
	availableObjs map[string]rpDesc[T]
	objStubs      []*rpStub[T]
	mu            sync.Mutex
}

func NewResourcePool[T Closeable](totalResource int, objFactories map[string]rpDesc[T]) (*ResourcePool[T], error) {
	if totalResource < 1 {
		// TODO: better error message using reflection
		return nil, fmt.Errorf("total memory must be greater than 0")
	}

	pool := ResourcePool[T]{
		totalResource: totalResource,
		objStubs:      []*rpStub[T]{},
		availableObjs: objFactories,
	}
	return &pool, nil
}

func (p *ResourcePool[T]) GetObj(ctx context.Context, name string) (T, error) {
	var obj T

	p.mu.Lock()
	defer p.mu.Unlock()

	objDesc, ok := p.availableObjs[name]
	if !ok {
		return obj, fmt.Errorf("obj %s not found", name)
	}

	existingObjStubIndex := -1
	for i, objStub := range p.objStubs {
		if objStub.name == name {
			if objStub.mu.TryLock() {
				existingObjStubIndex = i
				break
			}
		}
	}
	if existingObjStubIndex != -1 {
		// lock acquired, return this obj
		objStub := p.objStubs[existingObjStubIndex]
		// move to top of list
		p.objStubs = append(
			[]*rpStub[T]{objStub},
			append(
				p.objStubs[:existingObjStubIndex],
				p.objStubs[existingObjStubIndex+1:]...,
			)...,
		)
		if ctx.Err() != nil {
			// context already cancelled, return error
			objStub.mu.Unlock()
			return obj, ctx.Err()
		}
		go func() {
			defer objStub.mu.Unlock()
			<-ctx.Done()
		}()
		return objStub.obj, nil
	}
	// otherwise, try to create a new obj

	// calculate memory in use
	resourceInUse := 0
	for _, objStub := range p.objStubs {
		resourceInUse += objStub.resourceUsage
	}

	for {
		if resourceInUse+objDesc.resourceRequired <= p.totalResource {
			// condition met, create new obj
			break
		}
		if len(p.objStubs) == 0 {
			// no obj for free, no way to create new obj
			return obj, fmt.Errorf("not enough resource to create obj %s", name)
		}
		// free least recently used (last one in list)
		objStub := p.objStubs[len(p.objStubs)-1]
		objStub.mu.Lock()
		resourceInUse -= objStub.resourceUsage
		// remove from list
		p.objStubs = p.objStubs[:len(p.objStubs)-1]
		// close the obj
		objStub.obj.Close()
		objStub.mu.Unlock()
	}

	obj, err := objDesc.objFactory()
	if err != nil {
		return obj, fmt.Errorf("failed to create obj %s: %v", name, err)
	}
	newStub := &rpStub[T]{
		name:          name,
		resourceUsage: objDesc.resourceRequired,
		obj:           obj,
	}
	newStub.mu.Lock()
	if ctx.Err() != nil {
		// context already cancelled, return error
		newStub.mu.Unlock()
		return obj, ctx.Err()
	}
	go func() {
		defer newStub.mu.Unlock()
		<-ctx.Done()
	}()
	p.objStubs = append([]*rpStub[T]{newStub}, p.objStubs...)
	return obj, nil
}
