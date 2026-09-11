package downloader

import (
	"context"
	"fmt"
)

type githubAsset struct {
	name  string
	repo  string
	ref   string
	dest  string
	files []fileSpec
}

func newGitHubAsset(spec assetSpec) (Asset, error) {
	if spec.GitHub == nil || spec.GitHub.Repo == "" {
		return nil, fmt.Errorf("needs github.repo")
	}
	if spec.GitHub.Ref == "" {
		return nil, fmt.Errorf("needs github.ref")
	}
	if len(spec.GitHub.Files) == 0 {
		return nil, fmt.Errorf("needs at least one file")
	}

	return &githubAsset{
		name:  spec.Name,
		repo:  spec.GitHub.Repo,
		ref:   spec.GitHub.Ref,
		dest:  spec.Dest,
		files: spec.GitHub.Files,
	}, nil
}

func (a *githubAsset) Name() string { return a.name }

func (a *githubAsset) Version() string { return "" }

func (a *githubAsset) Tags() []string { return tagsOf(a.files) }

func (a *githubAsset) Paths(o *Options, sel Selection) []string {
	files := selectFiles(a.files, sel.Tags)

	paths := make([]string, 0, len(files))
	for _, file := range files {
		paths = append(paths, o.modelsPath(a.dest, file.Dest))
	}
	return paths
}

func (a *githubAsset) Fetch(ctx context.Context, o *Options, sel Selection) error {
	o.pickSource(o.Source, []string{SourceGitHub}, a.name)

	client := newHTTPClient()
	for _, file := range selectFiles(a.files, sel.Tags) {
		url := fmt.Sprintf("https://raw.githubusercontent.com/%s/%s/%s", a.repo, a.ref, file.Path)
		dest := o.modelsPath(a.dest, file.Dest)
		if err := downloadTo(ctx, client, url, dest, o.Force, o.log); err != nil {
			return err
		}
	}
	return nil
}
