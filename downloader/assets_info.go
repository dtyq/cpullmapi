package downloader

import (
	_ "embed"
	"fmt"

	"gopkg.in/yaml.v3"
)

//go:embed assets.yml
var assetsYAML []byte

type assetSpec struct {
	Name    string      `yaml:"name"`
	Kind    string      `yaml:"kind"`
	Repo    string      `yaml:"repo"`
	Sources []string    `yaml:"sources"`
	Dest    string      `yaml:"dest"`
	Version string      `yaml:"version"`
	Files   []fileSpec  `yaml:"files"`
	GitHub  *githubSpec `yaml:"github"`
	HF      *hfSpec     `yaml:"hf"`
}

// fileSpec 是仓库里的一个文件。写成裸字符串时本地路径与仓库路径相同。
type fileSpec struct {
	Path string
	Dest string
	// Tag 是筛选用的标签，比如 fp16、q8_0。空的表示这个文件不可筛，永远都要。
	Tag string
}

func (f *fileSpec) UnmarshalYAML(value *yaml.Node) error {
	if value.Kind == yaml.ScalarNode {
		f.Path, f.Dest = value.Value, value.Value
		return nil
	}

	var tmp struct {
		Path string `yaml:"path"`
		Dest string `yaml:"dest"`
		Tag  string `yaml:"tag"`
	}
	if err := value.Decode(&tmp); err != nil {
		return err
	}
	if tmp.Path == "" {
		return fmt.Errorf("file entry needs a path")
	}

	f.Path, f.Dest, f.Tag = tmp.Path, tmp.Dest, tmp.Tag
	if f.Dest == "" {
		f.Dest = f.Path
	}
	return nil
}

type githubSpec struct {
	Repo    string     `yaml:"repo"`
	Release string     `yaml:"release"`
	Archive string     `yaml:"archive"`
	Ref     string     `yaml:"ref"`
	Files   []fileSpec `yaml:"files"`
}

type hfSpec struct {
	Repo string `yaml:"repo"`
}

func parseAssets() ([]assetSpec, error) {
	var doc struct {
		Assets []assetSpec `yaml:"assets"`
	}
	if err := yaml.Unmarshal(assetsYAML, &doc); err != nil {
		return nil, fmt.Errorf("failed to parse assets.yml: %w", err)
	}

	seen := make(map[string]bool, len(doc.Assets))
	for _, spec := range doc.Assets {
		if spec.Name == "" {
			return nil, fmt.Errorf("asset without a name")
		}
		if seen[spec.Name] {
			return nil, fmt.Errorf("duplicate asset name %q", spec.Name)
		}
		seen[spec.Name] = true
	}

	return doc.Assets, nil
}
