package downloader

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
)

type hfAsset struct {
	name    string
	repo    string
	sources []string
	dest    string
	files   []fileSpec
}

func newHFAsset(spec assetSpec) (Asset, error) {
	if spec.Repo == "" {
		return nil, fmt.Errorf("needs a repo")
	}
	if len(spec.Files) == 0 {
		return nil, fmt.Errorf("needs at least one file")
	}

	return &hfAsset{
		name:    spec.Name,
		repo:    spec.Repo,
		sources: sourcesOrDefault(spec.Sources),
		dest:    spec.Dest,
		files:   spec.Files,
	}, nil
}

func (a *hfAsset) Name() string { return a.name }

func (a *hfAsset) Version() string { return "" }

func (a *hfAsset) Tags() []string { return tagsOf(a.files) }

func (a *hfAsset) Paths(o *Options, sel Selection) []string {
	files := selectFiles(a.files, sel.Tags)

	paths := make([]string, 0, len(files))
	for _, file := range files {
		paths = append(paths, o.modelsPath(a.dest, file.Dest))
	}
	return paths
}

func (a *hfAsset) Fetch(ctx context.Context, o *Options, sel Selection) error {
	source := o.pickSource(o.Source, a.sources, a.name)
	client := newHTTPClient()

	for _, file := range selectFiles(a.files, sel.Tags) {
		url := a.url(o, source, file.Path)
		dest := o.modelsPath(a.dest, file.Dest)
		if err := downloadTo(ctx, client, url, dest, o.Force, o.log); err != nil {
			return err
		}
	}
	return nil
}

func (a *hfAsset) url(o *Options, source, file string) string {
	if source == SourceModelScope {
		return fmt.Sprintf("%s/api/v1/models/%s/repo?Revision=master&FilePath=%s",
			o.MSEndpoint, a.repo, file)
	}
	return fmt.Sprintf("%s/%s/resolve/main/%s", o.HFEndpoint, a.repo, file)
}

// listHFRepo 列出仓库里的文件，用来下那些文件太多、不适合写进 assets.yml 的仓库。
func listHFRepo(ctx context.Context, client *http.Client, endpoint, repo string) ([]string, error) {
	url := fmt.Sprintf("%s/api/models/%s?blobs=true", endpoint, repo)

	body, _, err := get(ctx, client, url)
	if err != nil {
		return nil, err
	}
	defer body.Close()

	var info struct {
		Siblings []struct {
			Filename string `json:"rfilename"`
		} `json:"siblings"`
	}
	if err := json.NewDecoder(body).Decode(&info); err != nil {
		return nil, fmt.Errorf("failed to list %s: %w", repo, err)
	}

	files := make([]string, 0, len(info.Siblings))
	for _, sibling := range info.Siblings {
		files = append(files, sibling.Filename)
	}
	if len(files) == 0 {
		return nil, fmt.Errorf("%s has no files", repo)
	}
	return files, nil
}
