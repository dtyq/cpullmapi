//go:build with_image

package cpullmapi

import (
	"context"

	"github.com/davidbyttow/govips/v2/vips"
)

type ImageSegmentationInferencer interface {
	Inferencer
	SegmentImage(ctx context.Context, image *vips.ImageRef) ([]*vips.ImageRef, error)
}
