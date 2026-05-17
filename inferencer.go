package cpullmapi

import (
	"context"

	"github.com/davidbyttow/govips/v2/vips"
)

type Capability string

const (
	CapabilityImageSegmentation Capability = "image-seg"
	CapabilityOfflineASR        Capability = "offline-asr"
)

type ASRResult struct {
	Text    string `json:"text"`
	Lang    string `json:"lang"`
	Emotion string `json:"emotion"`
}

type Inferencer interface {
	GetCapabilities() []Capability

	SegmentImage(ctx context.Context, image *vips.ImageRef) ([]*vips.ImageRef, error)

	Transcribe(ctx context.Context, samples []float32) (result ASRResult, err error)

	Close()
}

type imagePostprocessFunc[T any] func(in *T, outputArray []float32) ([]*vips.ImageRef, error)

var InferencerFactoryMap = map[string]any /*func(config [T any]) (Inferencer, error)*/ {}

type DummyInferencer struct {
}

func (i *DummyInferencer) GetCapabilities() []Capability {
	return []Capability{}
}

func (i *DummyInferencer) SegmentImage(ctx context.Context, image *vips.ImageRef) ([]*vips.ImageRef, error) {
	return nil, ErrNotImplemented
}

func (i *DummyInferencer) Transcribe(ctx context.Context, samples []float32) (result ASRResult, err error) {
	return ASRResult{}, ErrNotImplemented
}

func (i *DummyInferencer) Close() {
}
