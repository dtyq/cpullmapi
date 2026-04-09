package cpullmapi

import (
	"context"
	"fmt"
	mathRand "math/rand"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	ort "github.com/yalue/onnxruntime_go"
)

func TestExecutorPool(t *testing.T) {
	pool, err := NewExecutorPool(2)
	if err != nil {
		t.Fatalf("failed to create executor pool: %v", err)
	}

	err = pool.Start(2)
	if err != nil {
		t.Fatalf("failed to start executor pool: %v", err)
	}
	defer pool.Stop()

	ort.SetSharedLibraryPath(onnxSharedLibraryPath)

	err = ort.InitializeEnvironment()
	if err != nil {
		t.Fatalf("failed to initialize ort environment: %v", err)
	}
	defer ort.DestroyEnvironment()

	var wg sync.WaitGroup
	wg.Add(1)
	testFunc := func() {
		defer wg.Done()

		var inferencer Inferencer
		inferencer, err = NewONNXBiRefNetInferencer(
			ONNXSODCommonConfig{
				ModelPath:              "./models/onnx-community/BiRefNet-ONNX/onnx/model_fp16.onnx",
				PreprocessorConfigPath: "./models/onnx-community/BiRefNet-ONNX/preprocessor_config.json",
			},
		)
		if err != nil {
			t.Fatalf("failed to create onnx inferencer: %v", err)
		}
		defer inferencer.Close()

		image, err := openImage("test/testphoto.jpg")
		if err != nil {
			t.Fatalf("failed to open image: %v", err)
		}

		segments, err := inferencer.SegmentImage(context.Background(), image)
		if err != nil {
			t.Fatalf("failed to segment image: %v", err)
		}

		for _, segment := range segments {
			_, _, err := segment.ExportPng(nil)
			if err != nil {
				t.Fatalf("failed to export segment: %v", err)
			}
		}
	}

	pool.Dispatch(testFunc)
	wg.Wait()
}

type testObjForRPool struct {
	resourceUsage int
	randomValue   int
	closed        bool
}

func (o *testObjForRPool) Close() {
	o.closed = true
}

func TestResourcePool(t *testing.T) {
	pool, err := NewResourcePool(8, map[string]RPDesc[*testObjForRPool]{
		"kind1": {
			ResourceRequired: 1,
			ObjFactory: func() (*testObjForRPool, error) {
				return &testObjForRPool{resourceUsage: 1, randomValue: mathRand.Intn(100)}, nil
			},
		},
		"kind4": {
			ResourceRequired: 4,
			ObjFactory: func() (*testObjForRPool, error) {
				return &testObjForRPool{resourceUsage: 4, randomValue: mathRand.Intn(100)}, nil
			},
		},
		"kind8": {
			ResourceRequired: 8,
			ObjFactory: func() (*testObjForRPool, error) {
				return &testObjForRPool{resourceUsage: 8, randomValue: mathRand.Intn(100)}, nil
			},
		},
		"kind9": {
			ResourceRequired: 9,
			ObjFactory: func() (*testObjForRPool, error) {
				return &testObjForRPool{resourceUsage: 9, randomValue: mathRand.Intn(100)}, nil
			},
		},
		"kindfailure": {
			ResourceRequired: 1,
			ObjFactory: func() (*testObjForRPool, error) {
				return nil, fmt.Errorf("failed to create obj")
			},
		},
	})
	if err != nil {
		t.Fatalf("failed to create resource pool: %v", err)
	}

	// errors - no such obj
	_, err = pool.GetObj(context.Background(), "kindnotexist")
	assert.ErrorContains(t, err, "obj kindnotexist not found")
	// errors - obj factory failed
	_, err = pool.GetObj(context.Background(), "kindfailure")
	assert.ErrorContains(t, err, "failed to create obj")
	// errors - not enough resource
	_, err = pool.GetObj(context.Background(), "kind9")
	assert.ErrorContains(t, err, "not enough resource to create obj kind9")
	// errors - context is already cancelled
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = pool.GetObj(ctx, "kind1")
	assert.ErrorIs(t, err, context.Canceled)

	// user1 - use kind1 and ended
	var kind1obj1 *testObjForRPool
	ctx1, cancel1 := context.WithCancel(context.Background())
	kind1obj1, err = pool.GetObj(ctx1, "kind1")
	assert.NoError(t, err)
	assert.NotNil(t, kind1obj1)
	assert.Equal(t, 1, kind1obj1.resourceUsage)

	// wait for user1 to end
	cancel1()
	time.Sleep(100 * time.Millisecond) // dummy wait for lock to be released

	// user2 - use kind1, should reuse
	ctx2, cancel2 := context.WithCancel(context.Background())
	obj, err := pool.GetObj(ctx2, "kind1")
	assert.NoError(t, err)
	assert.NotNil(t, obj)
	assert.Same(t, kind1obj1, obj)

	// no wait for user2 to end, kind1obj1 should still be used

	// user3 - use kind1, should create a new obj
	ctx3, cancel3 := context.WithCancel(context.Background())
	kind1obj2, err := pool.GetObj(ctx3, "kind1")
	assert.NoError(t, err)
	assert.NotNil(t, obj)
	assert.Equal(t, 1, obj.resourceUsage)
	assert.NotSame(t, kind1obj1, kind1obj2)

	// no wait for user3 to end, kind1obj2 should still be used

	// user4 - use kind4, should create a new obj
	ctx4, cancel4 := context.WithCancel(context.Background())
	kind4obj1, err := pool.GetObj(ctx4, "kind4")
	assert.NoError(t, err)
	assert.NotNil(t, kind4obj1)
	assert.Equal(t, 4, kind4obj1.resourceUsage)

	// close things
	cancel3()
	cancel2()
	time.Sleep(100 * time.Millisecond) // dummy wait for lock to be released

	// objs should not be kept
	assert.False(t, kind4obj1.closed)
	assert.False(t, kind1obj2.closed)
	assert.False(t, kind1obj1.closed)

	// at this point, the queue is [kind4obj1(running), kind1obj2(idle), kind1obj1(idle)]
	// user5 - use kind4, should create a new obj and evict kind1obj1, kind1obj2
	ctx5, cancel5 := context.WithCancel(context.Background())
	kind4obj2, err := pool.GetObj(ctx5, "kind4")
	assert.NoError(t, err)
	assert.NotNil(t, kind4obj2)
	assert.Equal(t, 4, kind4obj2.resourceUsage)
	assert.NotSame(t, kind4obj1, kind4obj2)

	cancel4()
	cancel5()
	time.Sleep(100 * time.Millisecond) // dummy wait for lock to be released

	// kind4 should be kept and kind1 should be evicted and closed
	assert.False(t, kind4obj1.closed)
	assert.False(t, kind4obj2.closed)
	assert.True(t, kind1obj2.closed)
	assert.True(t, kind1obj1.closed)

	// at this point, the queue is [kind4obj2(idle), kind4obj1(idle)]
	// user6 - use kind1, should create a new obj and evict kind4obj1
	ctx6, cancel6 := context.WithCancel(context.Background())
	kind1obj3, err := pool.GetObj(ctx6, "kind1")
	assert.NoError(t, err)
	assert.NotNil(t, kind1obj3)
	assert.Equal(t, 1, kind1obj3.resourceUsage)
	assert.NotSame(t, kind1obj1, kind1obj3)
	assert.NotSame(t, kind1obj2, kind1obj3)

	cancel6()
	time.Sleep(100 * time.Millisecond) // dummy wait for lock to be released

	// at this point, the queue is [kind1obj3(idle), kind4obj2(idle)]
	// user7 - use kind8, should create a new obj and evict others
	ctx7, cancel7 := context.WithCancel(context.Background())
	kind8obj1, err := pool.GetObj(ctx7, "kind8")
	assert.NoError(t, err)
	assert.NotNil(t, kind8obj1)
	assert.Equal(t, 8, kind8obj1.resourceUsage)

	cancel7()
}
