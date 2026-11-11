// Pure numerical helpers shared by the worker and browser verification page.
export const normalizeText = text => text.replace(/\s+/g, ' ').trim();

export function cosineScores(vector, embeddings, dimensions = 384) {
  if (vector.length !== dimensions || embeddings.length % dimensions) throw new Error('Dimensi embedding tidak cocok.');
  const count = embeddings.length / dimensions;
  const scores = new Float64Array(count);
  for (let row = 0; row < count; row++) {
    let dot = 0;
    for (let dim = 0; dim < dimensions; dim++) dot += vector[dim] * embeddings[row * dimensions + dim];
    scores[row] = dot;
  }
  return scores;
}

export function rankScores(scores) {
  return Array.from(scores, (score, index) => ({ index, score }))
    .sort((a, b) => b.score - a.score || a.index - b.index);
}

export function scoreHistogram(scores, bins = 24) {
  const minimum = Math.min(...scores), maximum = Math.max(...scores);
  const width = (maximum - minimum || 1) / bins;
  const counts = new Array(bins).fill(0);
  for (const value of scores) counts[Math.min(bins - 1, Math.floor((value - minimum) / width))]++;
  return { minimum, maximum, counts, mean: scores.reduce((a,b) => a+b, 0) / scores.length };
}

export function cosineBreakdown(query, embeddings, row, dimensions = 384) {
  const values = Array.from(query, (q, dim) => ({ dim, query: q,
    document: embeddings[row * dimensions + dim], product: q * embeddings[row * dimensions + dim] }));
  const score = values.reduce((sum, d) => sum + d.product, 0);
  const largest = [...values].sort((a,b) => Math.abs(b.product) - Math.abs(a.product)).slice(0, 8);
  return { score, largest, rest: score - largest.reduce((sum,d) => sum+d.product, 0),
    documentNorm: Math.sqrt(values.reduce((sum,d) => sum + d.document*d.document, 0)) };
}
