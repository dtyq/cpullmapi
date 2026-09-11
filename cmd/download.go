package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/spf13/cobra"

	"github.com/dtyq/cpullmapi/downloader"
)

type downloadFlags struct {
	source       string
	sherpaSource string
	hfEndpoint   string
	msEndpoint   string
	modelsDir    string
	libsDir      string
	force        bool
	list         bool
}

func newDownloadCommand() *cobra.Command {
	flags := &downloadFlags{}

	cmd := &cobra.Command{
		Use:   "download [name[@version][:tag...] ...]",
		Short: "Download models and the inference runtime",
		Long: "Download models and the inference runtime.\n\n" +
			"With no argument everything is downloaded. To fetch only some of them,\n" +
			"pass their names; `download --list` prints the names and, after a colon,\n" +
			"the tags each one can be narrowed down by.",
		RunE: func(cmd *cobra.Command, args []string) error {
			return runDownload(cmd, flags, args)
		},
	}

	f := cmd.Flags()
	f.StringVar(&flags.source, "source", downloader.SourceHuggingFace, "where to fetch models from: huggingface or modelscope")
	f.StringVar(&flags.sherpaSource, "sherpa-source", "", "where to fetch sherpa models from: github, huggingface or modelscope (defaults to --source)")
	f.StringVar(&flags.hfEndpoint, "hf-endpoint", envOr("HF_ENDPOINT", "https://huggingface.co"), "huggingface endpoint, point it at a mirror if needed")
	f.StringVar(&flags.msEndpoint, "ms-endpoint", "https://www.modelscope.cn", "modelscope endpoint")
	f.StringVar(&flags.modelsDir, "models-dir", "./models", "directory to store models in")
	f.StringVar(&flags.libsDir, "libs-dir", "./libs", "directory to store runtime libraries in")
	f.BoolVar(&flags.force, "force", false, "download again even if the files are already there")
	f.BoolVar(&flags.list, "list", false, "only list the assets and whether they are already local")

	return cmd
}

func runDownload(cmd *cobra.Command, flags *downloadFlags, args []string) error {
	if err := validateSources(flags); err != nil {
		return err
	}

	opts := &downloader.Options{
		Source:       flags.source,
		SherpaSource: flags.sherpaSource,
		HFEndpoint:   strings.TrimRight(flags.hfEndpoint, "/"),
		MSEndpoint:   strings.TrimRight(flags.msEndpoint, "/"),
		ModelsDir:    flags.modelsDir,
		LibsDir:      flags.libsDir,
		Force:        flags.force,
		Logf: func(format string, args ...any) {
			fmt.Fprintf(cmd.OutOrStdout(), format+"\n", args...)
		},
		Warnf: func(format string, args ...any) {
			fmt.Fprintf(cmd.ErrOrStderr(), format+"\n", args...)
		},
	}
	if opts.SherpaSource == "" {
		opts.SherpaSource = opts.Source
	}

	assets, err := downloader.Load()
	if err != nil {
		return err
	}

	selected, err := selectAssets(assets, args, opts)
	if err != nil {
		return err
	}

	if flags.list {
		for _, item := range selected {
			state := "missing"
			if downloader.Ready(item.asset, opts, item.sel) {
				state = "ready"
			}
			fmt.Fprintf(cmd.OutOrStdout(), "%-44s %-8s %s\n", item.asset.Name(), state, formatOptions(item.asset))
		}
		return nil
	}

	for _, item := range selected {
		fmt.Fprintf(cmd.OutOrStdout(), "==> %s\n", item.asset.Name())
		if err := item.asset.Fetch(cmd.Context(), opts, item.sel); err != nil {
			return fmt.Errorf("%s: %w", item.asset.Name(), err)
		}
	}
	return nil
}

// formatOptions 用命令行那套写法展示资产自带的版本和可选标签。
func formatOptions(asset downloader.Asset) string {
	var b strings.Builder

	if version := asset.Version(); version != "" {
		b.WriteString("@" + version)
	}
	for _, tag := range asset.Tags() {
		b.WriteString(" :" + tag)
	}
	return b.String()
}

// selection 是一条资产加上用户挑的版本与类型。
type selection struct {
	asset downloader.Asset
	sel   downloader.Selection
}

func selectAssets(assets []downloader.Asset, args []string, opts *downloader.Options) ([]selection, error) {
	if len(args) == 0 {
		return selectAll(assets), nil
	}

	byName := make(map[string]downloader.Asset, len(assets))
	for _, asset := range assets {
		byName[asset.Name()] = asset
	}

	selected := make([]selection, 0, len(args))
	for _, arg := range args {
		name, sel := parseArg(arg)

		asset, ok := byName[name]
		if !ok {
			return nil, fmt.Errorf("no asset named %q, `download --list` shows what there is", name)
		}

		sel, err := opts.CheckSelection(asset, sel)
		if err != nil {
			return nil, err
		}

		selected = append(selected, selection{asset: asset, sel: sel})
	}
	return selected, nil
}

func selectAll(assets []downloader.Asset) []selection {
	selected := make([]selection, 0, len(assets))
	for _, asset := range assets {
		selected = append(selected, selection{asset: asset})
	}
	return selected
}

// parseArg 拆开 名字[@版本][:标签[:标签...]]。
func parseArg(arg string) (string, downloader.Selection) {
	head, rest, _ := strings.Cut(arg, ":")
	name, version, _ := strings.Cut(head, "@")

	sel := downloader.Selection{Version: version}
	if rest != "" {
		sel.Tags = strings.Split(rest, ":")
	}
	return name, sel
}

func validateSources(flags *downloadFlags) error {
	switch flags.source {
	case downloader.SourceHuggingFace, downloader.SourceModelScope:
	default:
		return fmt.Errorf("--source must be %s or %s", downloader.SourceHuggingFace, downloader.SourceModelScope)
	}

	if flags.sherpaSource == "" {
		return nil
	}
	switch flags.sherpaSource {
	case downloader.SourceGitHub, downloader.SourceHuggingFace, downloader.SourceModelScope:
		return nil
	}
	return fmt.Errorf("--sherpa-source must be %s, %s or %s",
		downloader.SourceGitHub, downloader.SourceHuggingFace, downloader.SourceModelScope)
}

func envOr(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}
