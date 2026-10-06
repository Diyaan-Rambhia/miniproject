// Mutable values read by the 3D render loops every frame (kept out of React state on purpose).
export const store = {
  mx: 0,          // mouse -1..1
  my: 0,
  pipe: 0,        // pipeline scene progress 0..4 (float)
  heroOn: true,   // hero canvas visible
  pipeOn: false,  // pipeline canvas visible
  threat: 0.25,   // recent average threat 0..1, drives red packet ratio
  score: 0,       // score shown at the end of the pipeline story 0..100
  reduce: typeof window !== 'undefined' && window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches,
};
