package cpullmapi

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"gopkg.in/yaml.v3"
)

const mvanetModelDir = "models/onnx-community/MVANet-ONNX"

func requireTestAssets(t *testing.T, paths ...string) {
	t.Helper()

	// 建 ONNX 推理器离不开运行时库，缺了就直接跳过。
	for _, path := range append([]string{"libs/onnxruntime/lib/libonnxruntime.so"}, paths...) {
		if _, err := os.Stat(path); err != nil {
			t.Skipf("skipping: %s not found", path)
		}
	}
}

func TestHTTPConfig(t *testing.T) {
	emptyTokenHash := [48]byte{}
	tcs := []struct {
		name         string
		yaml         string
		expectErr    string
		expectConfig HTTPConfig
	}{
		{
			name: "NoTokenHash",
			yaml: `
bind: ":8080"
`,
			expectErr: "token hash must be 48 bytes binary base64ed",
		},
		{
			name: "InvalidTokenHash",
			yaml: `
bind: ":8080"
tokenHash: "1234567890"
`,
			expectErr: "failed to decode token hash",
		},
		{
			name: "ValidTokenHash",
			yaml: `
bind: ":8080"
tokenHash: AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
`,
			expectConfig: HTTPConfig{
				Bind:      ":8080",
				TokenHash: emptyTokenHash[:],
			},
		},
	}

	for _, tc := range tcs {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			var config HTTPConfig
			err := yaml.Unmarshal([]byte(tc.yaml), &config)
			if tc.expectErr != "" {
				assert.ErrorContains(t, err, tc.expectErr)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tc.expectConfig, config)
			}
		})
	}
}

func TestModelConfig(t *testing.T) {
	// cleanup := initORT(t)
	// defer cleanup()
	tmpDir := t.TempDir()

	tcs := []struct {
		name         string
		yaml         string
		expectErr    string
		expectConfig ModelConfig
		assets       []string
	}{
		{
			name: "NoModelType",
			yaml: `
requiredMemory: 1024
`,
			expectErr: "unknown model type:",
		},
		{
			name: "BadFactoryConfigType",
			yaml: `
requiredMemory: 1024
modelType: ONNXMVANet
factoryConfig: 123
`,
			expectErr: "failed to decode factory config for ONNXMVANet",
		},
		{
			name: "NoFactoryConfig",
			yaml: `
requiredMemory: 1024
modelType: ONNXMVANet
factoryConfig:
    cafebabe: deadbeef
`,
			expectErr: "failed to decode factory config for ONNXMVANet",
		},
		{
			name: "NoFactoryConfig",
			yaml: `
requiredMemory: 1024
modelType: ONNXMVANet
factoryConfig:
    modelPath: ` + filepath.Join(tmpDir, "notexist1") + `
    preprocessorConfigPath: ` + filepath.Join(tmpDir, "notexist2") + `
`,
			expectErr: "failed to decode factory config for ONNXMVANet",
		},
		{
			name: "OK",
			yaml: `
requiredMemory: 1024
modelType: ONNXMVANet
factoryConfig:
    modelPath: ` + mvanetModelDir + `/onnx/model_fp16.onnx
    preprocessorConfigPath: ` + mvanetModelDir + `/preprocessor_config.json
`,
			expectConfig: ModelConfig{
				RequiredMemory: 1024,
			},
			assets: []string{
				mvanetModelDir + "/onnx/model_fp16.onnx",
				mvanetModelDir + "/preprocessor_config.json",
			},
		},
	}
	for _, tc := range tcs {
		t.Run(tc.name, func(t *testing.T) {
			// t.Parallel()
			requireTestAssets(t, tc.assets...)

			var config ModelConfig
			err := yaml.Unmarshal([]byte(tc.yaml), &config)
			if tc.expectErr != "" {
				assert.ErrorContains(t, err, tc.expectErr)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tc.expectConfig.RequiredMemory, config.RequiredMemory)
				if !assert.NotNil(t, config.Factory) {
					t.Fatalf("factory is nil")
				}
				inferencer, err := config.Factory()
				assert.NoError(t, err)
				assert.NotNil(t, inferencer)
				assert.Equal(t, []Capability{CapabilityImageSegmentation}, inferencer.GetCapabilities())
			}
		})
	}
}

func TestConfigCreateMemoryPool(t *testing.T) {
	requireTestAssets(t, mvanetModelDir+"/onnx/model_fp16.onnx", mvanetModelDir+"/preprocessor_config.json")

	var err error
	// cleanup := initORT(t)
	// defer cleanup()

	configStr := `
http:
    bind: ":8080"
    tokenHash: AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
inference:
    totalMemory: 10240
    models:
        ONNXMVANet:
            requiredMemory: 1024
            modelType: ONNXMVANet
            factoryConfig:
                modelPath: ` + mvanetModelDir + `/onnx/model_fp16.onnx
                preprocessorConfigPath: ` + mvanetModelDir + `/preprocessor_config.json
`

	var config Config
	err = yaml.Unmarshal([]byte(configStr), &config)
	if err != nil {
		t.Fatalf("failed to unmarshal config: %v", err)
	}

	memoryPool, err := config.CreateMemoryPool()
	if err != nil {
		t.Fatalf("failed to create memory pool: %v", err)
	}

	assert.NotNil(t, memoryPool)
	inferencer, err := memoryPool.GetObj(context.Background(), "ONNXMVANet")
	if err != nil {
		t.Fatalf("failed to get inferencer: %v", err)
	}
	assert.NotNil(t, inferencer)
	assert.Equal(t, []Capability{CapabilityImageSegmentation}, inferencer.GetCapabilities())
}
