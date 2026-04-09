package cpullmapi

import (
	"fmt"
	"reflect"
	"regexp"

	"github.com/davidbyttow/govips/v2/vips"
	ort "github.com/yalue/onnxruntime_go"
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

func (c Config) MiscInitialize() error {
	// initialize ort environment
	ort.SetSharedLibraryPath(c.Inference.ONNXSharedLibraryPath)

	err := ort.InitializeEnvironment()
	if err != nil {
		return fmt.Errorf("failed to initialize ort environment: %v", err)
	}

	// initialize vips
	vips.LoggingSettings(func(messageDomain string, messageLevel vips.LogLevel, message string) {
		// empty implementation to omit startup logs
	}, c.Inference.VIPSLoggingLevel)
	err = vips.Startup(&vips.Config{
		ConcurrencyLevel: c.Inference.ThreadsPerSlot,
	})
	if err != nil {
		return fmt.Errorf("failed to initialize vips: %v", err)
	}

	return nil
}

func (c Config) MiscShutdown() {
	// shutdown vips
	vips.Shutdown()

	// shutdown ort environment
	ort.DestroyEnvironment()
}
