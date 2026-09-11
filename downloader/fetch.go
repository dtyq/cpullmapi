package downloader

import (
	"archive/tar"
	"archive/zip"
	"compress/bzip2"
	"compress/gzip"
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

func newHTTPClient() *http.Client {
	return &http.Client{
		Transport: &http.Transport{
			Proxy:                 http.ProxyFromEnvironment,
			ResponseHeaderTimeout: 60 * time.Second,
			IdleConnTimeout:       90 * time.Second,
			MaxIdleConns:          16,
		},
	}
}

func get(ctx context.Context, client *http.Client, url string) (io.ReadCloser, int64, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, 0, err
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, 0, err
	}
	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		return nil, 0, fmt.Errorf("%s: %s", url, resp.Status)
	}
	return resp.Body, resp.ContentLength, nil
}

// downloadTo 把 url 存成 dest。先写 .part 再改名，中断时不会留下看着完整的半个文件。
func downloadTo(ctx context.Context, client *http.Client, url, dest string, force bool, logf func(string, ...any)) error {
	if !force {
		if info, err := os.Stat(dest); err == nil && info.Size() > 0 {
			logf("skip %s, already downloaded", dest)
			return nil
		}
	}
	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		return err
	}

	body, total, err := get(ctx, client, url)
	if err != nil {
		return err
	}
	defer body.Close()

	part := dest + ".part"
	file, err := os.Create(part)
	if err != nil {
		return err
	}

	written, copyErr := copyWithProgress(file, body, total, filepath.Base(dest), logf)
	closeErr := file.Close()
	if copyErr != nil || closeErr != nil {
		os.Remove(part)
		if copyErr != nil {
			return copyErr
		}
		return closeErr
	}
	if total > 0 && written != total {
		os.Remove(part)
		return fmt.Errorf("%s: got %d of %d bytes", url, written, total)
	}

	return os.Rename(part, dest)
}

func copyWithProgress(dst io.Writer, src io.Reader, total int64, label string, logf func(string, ...any)) (int64, error) {
	var written int64
	lastDecile := int64(-1)
	buf := make([]byte, 256*1024)

	for {
		n, readErr := src.Read(buf)
		if n > 0 {
			if _, err := dst.Write(buf[:n]); err != nil {
				return written, err
			}
			written += int64(n)

			if total > 0 {
				if decile := written * 10 / total; decile != lastDecile {
					lastDecile = decile
					logf("  %s: %d%%", label, decile*10)
				}
			}
		}
		if readErr == io.EOF {
			return written, nil
		}
		if readErr != nil {
			return written, readErr
		}
	}
}

// extractTarStream 把 tar 流解到 dest，丢掉前面 strip 层目录。
func extractTarStream(src io.Reader, dest string, strip int) error {
	reader := tar.NewReader(src)

	for {
		header, err := reader.Next()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}

		target, err := archiveTarget(dest, archiveRelative(header.Name, strip))
		if err != nil {
			return err
		}
		if target == "" {
			continue
		}

		switch header.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(target, 0o755); err != nil {
				return err
			}
		case tar.TypeReg:
			if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
				return err
			}
			if err := writeFile(target, reader); err != nil {
				return err
			}
		}
	}
}

// extractZipStream 要求能随机读，所以 zip 走 ReaderAt。
func extractZipStream(src io.ReaderAt, size int64, dest string, strip int) error {
	archive, err := zip.NewReader(src, size)
	if err != nil {
		return err
	}

	for _, entry := range archive.File {
		target, err := archiveTarget(dest, archiveRelative(entry.Name, strip))
		if err != nil {
			return err
		}
		if target == "" {
			continue
		}

		if entry.FileInfo().IsDir() {
			if err := os.MkdirAll(target, 0o755); err != nil {
				return err
			}
			continue
		}

		if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
			return err
		}

		rc, err := entry.Open()
		if err != nil {
			return err
		}
		writeErr := writeFile(target, rc)
		rc.Close()
		if writeErr != nil {
			return writeErr
		}
	}
	return nil
}

func decompress(name string, src io.Reader) (io.Reader, error) {
	switch {
	case strings.HasSuffix(name, ".tar.gz"), strings.HasSuffix(name, ".tgz"):
		return gzip.NewReader(src)
	case strings.HasSuffix(name, ".tar.bz2"):
		return bzip2.NewReader(src), nil
	default:
		return nil, fmt.Errorf("unsupported archive %q", name)
	}
}

func writeFile(path string, src io.Reader) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}

	_, copyErr := io.Copy(file, src)
	closeErr := file.Close()
	if copyErr != nil {
		return copyErr
	}
	return closeErr
}

// archiveRelative 去掉前 strip 层目录，给出条目在归档里的相对路径。
func archiveRelative(name string, strip int) string {
	parts := strings.Split(filepath.ToSlash(filepath.Clean(name)), "/")
	if len(parts) <= strip {
		return ""
	}
	return strings.Join(parts[strip:], "/")
}

// archiveTarget 把相对路径拼成解压目标，顺带挡掉 ../ 这类越界的条目。
func archiveTarget(dest, rel string) (string, error) {
	if rel == "" || rel == "." {
		return "", nil
	}

	target := filepath.Join(dest, filepath.FromSlash(rel))
	if !strings.HasPrefix(target, filepath.Clean(dest)+string(os.PathSeparator)) {
		return "", fmt.Errorf("archive entry %q points outside %s", rel, dest)
	}
	return target, nil
}
