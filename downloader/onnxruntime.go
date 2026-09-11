package downloader

import (
	"archive/tar"
	"archive/zip"
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

const (
	onnxRuntimeRepo   = "microsoft/onnxruntime"
	onnxRuntimeLibDir = "lib"
)

type onnxRuntimeAsset struct {
	name    string
	version string
}

func newONNXRuntimeAsset(spec assetSpec) (Asset, error) {
	if spec.Version == "" {
		return nil, fmt.Errorf("needs a version")
	}
	return &onnxRuntimeAsset{name: spec.Name, version: spec.Version}, nil
}

func (a *onnxRuntimeAsset) Name() string { return a.name }

func (a *onnxRuntimeAsset) Version() string { return a.version }

func (a *onnxRuntimeAsset) Tags() []string { return nil }

// archive 挑 release 里的 cpu 包。同一个 release 还挂着各种 gpu 变体，都不需要。
func (a *onnxRuntimeAsset) archive(version string) (string, error) {
	switch runtime.GOOS {
	case "linux":
		switch runtime.GOARCH {
		case "amd64":
			return fmt.Sprintf("onnxruntime-linux-x64-%s.tgz", version), nil
		case "arm64":
			return fmt.Sprintf("onnxruntime-linux-aarch64-%s.tgz", version), nil
		}
	case "darwin":
		if runtime.GOARCH == "arm64" {
			return fmt.Sprintf("onnxruntime-osx-arm64-%s.tgz", version), nil
		}
	case "windows":
		switch runtime.GOARCH {
		case "amd64":
			return fmt.Sprintf("onnxruntime-win-x64-%s.zip", version), nil
		case "arm64":
			return fmt.Sprintf("onnxruntime-win-arm64-%s.zip", version), nil
		}
	}
	return "", fmt.Errorf("onnxruntime ships no %s/%s build", runtime.GOOS, runtime.GOARCH)
}

func (a *onnxRuntimeAsset) libName() string {
	switch runtime.GOOS {
	case "windows":
		return "onnxruntime.dll"
	case "darwin":
		return "libonnxruntime.dylib"
	default:
		return "libonnxruntime.so"
	}
}

func (a *onnxRuntimeAsset) Paths(o *Options, _ Selection) []string {
	return []string{o.libsPath(a.name, onnxRuntimeLibDir, a.libName())}
}

func (a *onnxRuntimeAsset) Fetch(ctx context.Context, o *Options, sel Selection) error {
	version := a.version
	if sel.Version != "" {
		version = sel.Version
	}

	// 本地文件名不带版本，所以指定了别的版本就不能因为「文件在」而跳过。
	if !o.Force && version == a.version && pathsReady(a.Paths(o, sel)) {
		o.log("skip %s, already downloaded", a.Paths(o, sel)[0])
		return nil
	}

	archive, err := a.archive(version)
	if err != nil {
		return err
	}

	url := fmt.Sprintf("https://github.com/%s/releases/download/v%s/%s", onnxRuntimeRepo, version, archive)
	dest := o.libsPath(a.name)
	client := newHTTPClient()

	if strings.HasSuffix(archive, ".zip") {
		return a.fetchZip(ctx, o, client, url, dest)
	}
	return a.fetchTar(ctx, o, client, url, dest)
}

func (a *onnxRuntimeAsset) fetchTar(ctx context.Context, o *Options, client *http.Client, url, dest string) error {
	body, _, err := get(ctx, client, url)
	if err != nil {
		return err
	}
	defer body.Close()

	reader, err := decompress(url, body)
	if err != nil {
		return err
	}

	if err := os.MkdirAll(dest, 0o755); err != nil {
		return err
	}

	archive := tar.NewReader(reader)
	for {
		header, err := archive.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		// 包里只有 .so 是真文件，同名的 .so.1 之类是链接，跳过就够。
		if header.Typeflag != tar.TypeReg {
			continue
		}

		rel := archiveRelative(header.Name, 1)
		if !flatLibEntry(rel) {
			continue
		}

		target, err := archiveTarget(dest, rel)
		if err != nil {
			return err
		}
		if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
			return err
		}
		if err := writeFile(target, archive); err != nil {
			return err
		}
	}

	return a.linkVersionedLib(dest)
}

func (a *onnxRuntimeAsset) fetchZip(ctx context.Context, o *Options, client *http.Client, url, dest string) error {
	file, err := os.CreateTemp("", "onnxruntime-*.zip")
	if err != nil {
		return err
	}
	defer func() {
		file.Close()
		os.Remove(file.Name())
	}()

	body, total, err := get(ctx, client, url)
	if err != nil {
		return err
	}
	defer body.Close()

	if _, err := copyWithProgress(file, body, total, filepath.Base(url), o.log); err != nil {
		return err
	}

	info, err := file.Stat()
	if err != nil {
		return err
	}
	archive, err := zip.NewReader(file, info.Size())
	if err != nil {
		return err
	}

	if err := os.MkdirAll(dest, 0o755); err != nil {
		return err
	}

	for _, entry := range archive.File {
		if entry.FileInfo().IsDir() {
			continue
		}

		rel := archiveRelative(entry.Name, 1)
		if !flatLibEntry(rel) {
			continue
		}

		target, err := archiveTarget(dest, rel)
		if err != nil {
			return err
		}
		if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
			return err
		}

		reader, err := entry.Open()
		if err != nil {
			return err
		}
		writeErr := writeFile(target, reader)
		reader.Close()
		if writeErr != nil {
			return writeErr
		}
	}

	return a.linkVersionedLib(dest)
}

// flatLibEntry 只认 lib/ 下的直接文件，包里的 cmake/pkgconfig/include 一概不要。
func flatLibEntry(rel string) bool {
	rest, ok := strings.CutPrefix(rel, onnxRuntimeLibDir+"/")
	return ok && rest != "" && !strings.Contains(rest, "/")
}

// linkVersionedLib 补一个不带版本号的链接，省得配置里写死版本。
func (a *onnxRuntimeAsset) linkVersionedLib(dest string) error {
	if runtime.GOOS == "windows" {
		return nil
	}

	libDir := filepath.Join(dest, onnxRuntimeLibDir)

	suffix := ".so.*"
	if runtime.GOOS == "darwin" {
		suffix = ".*.dylib"
	}

	matches, err := filepath.Glob(filepath.Join(libDir, "libonnxruntime"+suffix))
	if err != nil || len(matches) == 0 {
		return err
	}

	link := filepath.Join(libDir, a.libName())
	os.Remove(link)
	return os.Symlink(filepath.Base(matches[0]), link)
}
