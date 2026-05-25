package cpullmapi

import (
	"fmt"
	"net"
	"reflect"
	"regexp"

	"github.com/davidbyttow/govips/v2/vips"
	"gopkg.in/yaml.v3"
)

type ModelConfig struct {
	RequiredMemory int                        `yaml:"requiredMemory"`
	Factory        func() (Inferencer, error) `yaml:"-"`
}

func (c *ModelConfig) UnmarshalYAML(value *yaml.Node) error {
	var tmp struct {
		RequiredMemory int       `yaml:"requiredMemory"`
		ModelType      string    `yaml:"modelType"`
		FactoryConfig  yaml.Node `yaml:"factoryConfig"`
	}

	if err := value.Decode(&tmp); err != nil {
		return fmt.Errorf("failed to decode model config: %v", err)
	}

	c.RequiredMemory = tmp.RequiredMemory

	inferencerFactory, ok := InferencerFactoryMap[tmp.ModelType]
	if !ok {
		return fmt.Errorf("unknown model type: %s", tmp.ModelType)
	}

	reflectFactory := reflect.ValueOf(inferencerFactory)
	if reflectFactory.Type().NumIn() != 1 {
		panic(
			fmt.Errorf(
				"inferencer factory must have exactly one argument, got %d for %s",
				reflectFactory.Type().NumIn(),
				tmp.ModelType,
			),
		)
	}
	if reflectFactory.Type().NumOut() != 2 {
		panic(
			fmt.Errorf(
				"inferencer factory must have exactly two return values, got %d for %s",
				reflectFactory.Type().NumOut(),
				tmp.ModelType,
			),
		)
	}

	argValue := reflect.New(reflectFactory.Type().In(0))
	err := tmp.FactoryConfig.Decode(argValue.Interface())
	if err != nil {
		return fmt.Errorf("failed to decode factory config for %s: %v", tmp.ModelType, err)
	}

	c.Factory = func() (Inferencer, error) {
		// fmt.Fprintf(os.Stderr, "calling factory for %s: %#+v\n", tmp.ModelType, argValue.Elem().Interface())
		results := reflectFactory.Call([]reflect.Value{argValue.Elem()})
		inferencer, _ := results[0].Interface().(Inferencer)
		resultErr, _ := results[1].Interface().(error)
		return inferencer, resultErr
	}
	return nil
}

type InferenceConfig struct {
	VIPSLoggingLevel      vips.LogLevel `yaml:"vipsLoggingLevel"`
	ONNXSharedLibraryPath string        `yaml:"onnxSharedLibraryPath"`

	TotalMemory int                    `yaml:"totalMemory"`
	Models      map[string]ModelConfig `yaml:"models"`

	SlotCount      int `yaml:"slotCount"`
	ThreadsPerSlot int `yaml:"threadsPerSlot"`
}

type IncomingConfig struct {
	ProxyRequestInspector bool         `yaml:"proxyRequestInspector"`
	AllowCIDR             []*net.IPNet `yaml:"allowCIDR"`
}

func (c *IncomingConfig) UnmarshalYAML(value *yaml.Node) error {
	var tmp struct {
		ProxyRequestInspector bool     `yaml:"proxyRequestInspector"`
		AllowCIDR             []string `yaml:"allowCIDR"`
	}
	if err := value.Decode(&tmp); err != nil {
		return err
	}

	c.AllowCIDR = make([]*net.IPNet, len(tmp.AllowCIDR))
	for i, cidr := range tmp.AllowCIDR {
		_, ipNet, err := net.ParseCIDR(cidr)
		if err != nil {
			return fmt.Errorf("failed to parse CIDR \"%s\": %v", cidr, err)
		}
		c.AllowCIDR[i] = ipNet
	}

	c.ProxyRequestInspector = tmp.ProxyRequestInspector
	return nil
}

type OutgoingConfig struct {
	AllowRegexp []regexp.Regexp `yaml:"allowRegexp"`
}

func (c *OutgoingConfig) UnmarshalYAML(value *yaml.Node) error {
	var tmp struct {
		AllowRegexp []string `yaml:"allowRegexp"`
	}
	if err := value.Decode(&tmp); err != nil {
		return err
	}
	for _, reStr := range tmp.AllowRegexp {
		re, err := regexp.Compile(reStr)
		if err != nil {
			return fmt.Errorf("failed to compile regexp \"%s\": %v", reStr, err)
		}
		c.AllowRegexp = append(c.AllowRegexp, *re)
	}
	return nil
}

type Config struct {
	HTTP     HTTPConfig        `yaml:"http"`
	Logs     []LogStreamConfig `yaml:"logs"`
	Incoming IncomingConfig    `yaml:"incoming"`
	Outgoing OutgoingConfig    `yaml:"outgoing"`

	Inference InferenceConfig `yaml:"inference"`
}

func ConfigFromYAML(data []byte) (*Config, error) {
	var config Config
	err := yaml.Unmarshal(data, &config)
	if err != nil {
		return nil, err
	}
	return &config, nil
}

func (c Config) CreateMemoryPool() (*ResourcePool[Inferencer], error) {
	// parse model configs
	inferencerDescs := make(map[string]RPDesc[Inferencer])
	for name, modelConfig := range c.Inference.Models {
		inferencerDescs[name] = RPDesc[Inferencer]{
			ResourceRequired: modelConfig.RequiredMemory,
			ObjFactory:       modelConfig.Factory,
		}
	}

	return NewResourcePool(c.Inference.TotalMemory, inferencerDescs)
}

var ConfigInitFuncs = []func(*Config) error{}

func (c Config) MiscInitialize() error {
	var err error
	for _, initFunc := range ConfigInitFuncs {
		err = initFunc(&c)
		if err != nil {
			return fmt.Errorf("failed to run config init func: %v", err)
		}
	}
	return nil
}

var ConfigShutdownFuncs = []func(*Config){}

func (c Config) MiscShutdown() {
	for _, shutdownFunc := range ConfigShutdownFuncs {
		shutdownFunc(&c)
	}
}
