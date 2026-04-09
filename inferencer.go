package cpullmapi

import (
	"context"
	"errors"

	"github.com/davidbyttow/govips/v2/vips"
)

type Capability string

const (
	CapabilityImageSegmentation Capability = "image-seg"
)

var ErrNotImplemented = errors.New("not implemented")

type Inferencer interface {
	GetCapabilities() []Capability

	SegmentImage(ctx context.Context, image *vips.ImageRef) ([]*vips.ImageRef, error)

	Close()
}

type imagePostprocessFunc[T any] func(in *T, outputArray []float32) ([]*vips.ImageRef, error)

var InferencerFactoryMap = map[string]any /*func(config [T any]) (Inferencer, error)*/ {}
