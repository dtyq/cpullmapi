package downloader

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
)

const (
	SourceHuggingFace = "huggingface"
	SourceModelScope  = "modelscope"
	SourceGitHub      = "github"
)

// Options 是下载时用户能调整的全部东西。
type Options struct {
	// Source 决定优先从哪个模型站取，huggingface 或 modelscope。
	Source string
	// SherpaSource 单独控制 sherpa 模型包的来源，可以选 github。
	SherpaSource string

	HFEndpoint string
	MSEndpoint string

	ModelsDir string
	LibsDir   string

	Force bool

	Logf  func(format string, args ...any)
	Warnf func(format string, args ...any)
}

// Selection 是从命令行 名字[@版本][:标签...] 里解析出来的选择。
type Selection struct {
	// Version 覆盖资产自带的版本，空表示用默认的。
	Version string
	// Tags 是筛选标签，空表示全要。
	Tags []string
}

// Asset 是一份可以下载的东西。各实现自己决定怎么取、放到哪。
type Asset interface {
	Name() string
	// Version 是资产自带的版本，空表示它没有版本的概念。
	Version() string
	// Tags 列出可用于筛选的标签。空的表示没得筛，不能写 :标签。
	Tags() []string
	// Paths 给出筛选之后本地应有的路径，用来判断是否已经就绪。
	Paths(o *Options, sel Selection) []string
	Fetch(ctx context.Context, o *Options, sel Selection) error
}

func Load() ([]Asset, error) {
	specs, err := parseAssets()
	if err != nil {
		return nil, err
	}

	assets := make([]Asset, 0, len(specs))
	for _, spec := range specs {
		asset, err := newAsset(spec)
		if err != nil {
			return nil, fmt.Errorf("asset %q: %w", spec.Name, err)
		}
		assets = append(assets, asset)
	}
	return assets, nil
}

func newAsset(spec assetSpec) (Asset, error) {
	switch spec.Kind {
	case "hf":
		return newHFAsset(spec)
	case "github":
		return newGitHubAsset(spec)
	case "sherpa":
		return newSherpaAsset(spec)
	case "onnxruntime":
		return newONNXRuntimeAsset(spec)
	default:
		return nil, fmt.Errorf("unknown kind %q", spec.Kind)
	}
}

func (o *Options) modelsPath(parts ...string) string {
	return filepath.Join(append([]string{o.ModelsDir}, parts...)...)
}

func (o *Options) libsPath(parts ...string) string {
	return filepath.Join(append([]string{o.LibsDir}, parts...)...)
}

func (o *Options) log(format string, args ...any) {
	if o.Logf != nil {
		o.Logf(format, args...)
	}
}

func (o *Options) warn(format string, args ...any) {
	if o.Warnf != nil {
		o.Warnf(format, args...)
	}
}

// Ready 说明选中的那部分在本地齐了。
func Ready(asset Asset, o *Options, sel Selection) bool {
	return pathsReady(asset.Paths(o, sel))
}

// selectFiles 按标签挑文件。tags 为空表示全要，没标签的文件永远都要。
func selectFiles(files []fileSpec, tags []string) []fileSpec {
	if len(tags) == 0 {
		return files
	}

	wanted := make(map[string]bool, len(tags))
	for _, tag := range tags {
		wanted[tag] = true
	}

	selected := make([]fileSpec, 0, len(files))
	for _, file := range files {
		if file.Tag == "" || wanted[file.Tag] {
			selected = append(selected, file)
		}
	}
	return selected
}

func tagsOf(files []fileSpec) []string {
	seen := make(map[string]bool, len(files))
	tags := make([]string, 0, len(files))

	for _, file := range files {
		if file.Tag != "" && !seen[file.Tag] {
			seen[file.Tag] = true
			tags = append(tags, file.Tag)
		}
	}

	sort.Strings(tags)
	return tags
}

// CheckSelection 校验版本和标签，顺手把用不上的那部分丢掉。
// 资产本身没有版本或者没有标签可筛时只是警告，写错标签才当成错误。
func (o *Options) CheckSelection(asset Asset, sel Selection) (Selection, error) {
	if sel.Version != "" && asset.Version() == "" {
		o.warn("warning: %s has no version, ignoring @%s", asset.Name(), sel.Version)
		sel.Version = ""
	}

	if len(sel.Tags) == 0 {
		return sel, nil
	}

	available := asset.Tags()
	if len(available) == 0 {
		o.warn("warning: %s has no tags, ignoring :%s", asset.Name(), strings.Join(sel.Tags, ":"))
		sel.Tags = nil
		return sel, nil
	}

	for _, tag := range sel.Tags {
		if !slices.Contains(available, tag) {
			return sel, fmt.Errorf("%s has no tag %q, available: %s", asset.Name(), tag, strings.Join(available, " "))
		}
	}
	return sel, nil
}

// pickSource 在候选源里挑一个。用户要的源用不了时回落到第一个候选并说明，
// 免得因为某个站没镜像就整个失败。
func (o *Options) pickSource(requested string, available []string, what string) string {
	if len(available) == 0 {
		return requested
	}
	for _, source := range available {
		if source == requested {
			return requested
		}
	}

	fallback := available[0]
	o.warn("warning: %s is not on %s, falling back to %s", what, requested, fallback)
	return fallback
}

// sourcesOrDefault 把声明的可用源补全。没写就认为两个模型站都行。
func sourcesOrDefault(declared []string) []string {
	if len(declared) > 0 {
		return declared
	}
	return []string{SourceHuggingFace, SourceModelScope}
}

func pathsReady(paths []string) bool {
	for _, path := range paths {
		info, err := os.Stat(path)
		if err != nil {
			return false
		}

		if info.IsDir() {
			entries, err := os.ReadDir(path)
			if err != nil || len(entries) == 0 {
				return false
			}
			continue
		}

		if info.Size() == 0 {
			return false
		}
	}
	return true
}
