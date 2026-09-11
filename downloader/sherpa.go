package downloader

import (
	"context"
	"fmt"
	"path/filepath"
	"strings"
)

const sherpaRepo = "k2-fsa/sherpa-onnx"

type sherpaAsset struct {
	name          string
	dest          string
	githubRelease string
	archive       string
	hfRepo        string
}

func newSherpaAsset(spec assetSpec) (Asset, error) {
	asset := &sherpaAsset{name: spec.Name, dest: spec.Dest}

	if spec.GitHub != nil {
		if spec.GitHub.Release == "" || spec.GitHub.Archive == "" {
			return nil, fmt.Errorf("needs github.release and github.archive")
		}
		asset.githubRelease = spec.GitHub.Release
		asset.archive = spec.GitHub.Archive
	}
	if spec.HF != nil {
		asset.hfRepo = spec.HF.Repo
	}

	if asset.githubRelease == "" && asset.hfRepo == "" {
		return nil, fmt.Errorf("needs a github release or an hf repo")
	}
	return asset, nil
}

func (a *sherpaAsset) Name() string { return a.name }

func (a *sherpaAsset) Version() string { return "" }

func (a *sherpaAsset) Tags() []string { return nil }

func (a *sherpaAsset) Paths(o *Options, _ Selection) []string {
	return []string{o.modelsPath(a.dest)}
}

func (a *sherpaAsset) Dests(o *Options) []string {
	return []string{o.modelsPath(a.dest)}
}

func (a *sherpaAsset) sources() []string {
	sources := make([]string, 0, 2)
	if a.githubRelease != "" {
		sources = append(sources, SourceGitHub)
	}
	if a.hfRepo != "" {
		sources = append(sources, SourceHuggingFace)
	}
	return sources
}

func (a *sherpaAsset) Fetch(ctx context.Context, o *Options, _ Selection) error {
	if o.pickSource(o.SherpaSource, a.sources(), a.name) == SourceGitHub {
		return a.fetchRelease(ctx, o)
	}
	return a.fetchRepo(ctx, o)
}

// fetchRelease 取 github release 里打好的 tar.bz2。包里带着模型同名的顶层目录，
// 所以剥掉一层正好落到 dest。
func (a *sherpaAsset) fetchRelease(ctx context.Context, o *Options) error {
	dest := o.modelsPath(a.dest)
	if !o.Force && pathsReady(a.Paths(o, Selection{})) {
		o.log("skip %s, already downloaded", dest)
		return nil
	}

	url := fmt.Sprintf("https://github.com/%s/releases/download/%s/%s", sherpaRepo, a.githubRelease, a.archive)

	client := newHTTPClient()
	body, _, err := get(ctx, client, url)
	if err != nil {
		return err
	}
	defer body.Close()

	reader, err := decompress(a.archive, body)
	if err != nil {
		return err
	}
	return extractTarStream(reader, dest, 1)
}

func (a *sherpaAsset) fetchRepo(ctx context.Context, o *Options) error {
	client := newHTTPClient()

	files, err := listHFRepo(ctx, client, o.HFEndpoint, a.hfRepo)
	if err != nil {
		return err
	}

	for _, file := range files {
		if strings.HasPrefix(filepath.Base(file), ".git") {
			continue
		}
		url := fmt.Sprintf("%s/%s/resolve/main/%s", o.HFEndpoint, a.hfRepo, file)
		dest := o.modelsPath(a.dest, file)
		if err := downloadTo(ctx, client, url, dest, o.Force, o.log); err != nil {
			return err
		}
	}
	return nil
}
