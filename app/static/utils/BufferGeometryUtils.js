// Minimal shim to satisfy GLTFLoader dependency.
// If your models require TriangleFan/Strip conversion, swap this with the real
// BufferGeometryUtils from Three.js examples (MIT) or implement conversion.
import { BufferGeometry } from 'three';

/**
 * No-op conversion: returns the input geometry as-is.
 * Replace with a full implementation if your assets use non-triangle draw modes.
 */
export function toTrianglesDrawMode(geometry, /* drawMode */) {
  // ensure a BufferGeometry instance is returned
  return geometry instanceof BufferGeometry ? geometry : new BufferGeometry().copy(geometry);
}

export default { toTrianglesDrawMode };
