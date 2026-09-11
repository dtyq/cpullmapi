package downloader

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// StageItem 是一条资产加上用户挑的版本与类型。
type StageItem struct {
	Asset     Asset
	Selection Selection
}

// Stage 把选中的资产复制一份到 stageRoot 下，按 models/ 和 libs/ 分开放，内部的
// 相对布局照搬。镜像构建之类只需要拷目录的场景可以直接拿这个当源。
//
// 没写标签就整目录拷：库旁边那些符号链接和带版本号的实体文件不在 Paths 里，
// 单独拷 Paths 会拷出一个断链。写了标签就只拷选中的文件，跟下载的行为对齐。
func Stage(stageRoot string, items []StageItem, o *Options) error {
	for _, item := range items {
		paths := item.Asset.Dests(o)
		if len(item.Selection.Tags) > 0 {
			paths = item.Asset.Paths(o, item.Selection)
		}

		for _, path := range paths {
			relative, ok := stageRelative(path, o)
			if !ok {
				return fmt.Errorf("%s: %s is outside the models and libs directories", item.Asset.Name(), path)
			}

			if _, err := os.Stat(path); err != nil {
				return fmt.Errorf("%s: %w", item.Asset.Name(), err)
			}

			target := filepath.Join(stageRoot, relative)
			o.log("stage %s -> %s", path, target)
			if err := copyTree(path, target); err != nil {
				return fmt.Errorf("%s: %w", item.Asset.Name(), err)
			}
		}
	}
	return nil
}

// stageRelative 把 models 或 libs 下的路径换成 stage 里的相对路径，不在两者之内的返回 false。
func stageRelative(path string, o *Options) (string, bool) {
	roots := []struct {
		dir  string
		name string
	}{
		{o.ModelsDir, "models"},
		{o.LibsDir, "libs"},
	}

	for _, root := range roots {
		// modelsPath/libsPath 走 filepath.Join，会把 ./models 洗成 models，这里跟着洗一遍。
		dir := filepath.Clean(root.dir)
		if sub, ok := strings.CutPrefix(path, dir+string(filepath.Separator)); ok {
			return filepath.Join(root.name, sub), true
		}
	}
	return "", false
}

// copyTree 复制文件或目录，保留权限和符号链接。
func copyTree(src, dst string) error {
	info, err := os.Lstat(src)
	if err != nil {
		return err
	}

	switch {
	case info.Mode()&os.ModeSymlink != 0:
		target, err := os.Readlink(src)
		if err != nil {
			return err
		}
		if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
			return err
		}
		if err := os.Remove(dst); err != nil && !os.IsNotExist(err) {
			return err
		}
		return os.Symlink(target, dst)

	case info.IsDir():
		if err := os.MkdirAll(dst, info.Mode().Perm()); err != nil {
			return err
		}
		entries, err := os.ReadDir(src)
		if err != nil {
			return err
		}
		for _, entry := range entries {
			if err := copyTree(filepath.Join(src, entry.Name()), filepath.Join(dst, entry.Name())); err != nil {
				return err
			}
		}
		return nil

	case info.Mode().IsRegular():
		return copyFile(src, dst, info.Mode().Perm())

	default:
		// 设备、fifo、socket 之类没什么好复制的。
		return nil
	}
}

func copyFile(src, dst string, perm os.FileMode) error {
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}

	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()

	out, err := os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, perm)
	if err != nil {
		return err
	}

	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		return err
	}
	return out.Close()
}
