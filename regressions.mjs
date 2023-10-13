export function linearRegression(logSeconds, watts) {
  let n = 0;
  let sum_x = 0;
  let sum_y = 0;
  let sum_xy = 0;
  let sum_xx = 0;
  let sum_yy = 0;

  for (let i = 0; i < watts.length; i++) {
    if(watts[i]) { // ignore 0 values
      n++;
      sum_x += logSeconds[i];
      sum_y += watts[i];
      sum_xy += logSeconds[i] * watts[i];
      sum_xx += logSeconds[i] * logSeconds[i];
      sum_yy += watts[i] * watts[i];
    }
  }

  return {
    slope: (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x),
    intercept: (sum_y - (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x) * sum_x) / n,
    r2: Math.pow((n * sum_xy - sum_x * sum_y) / Math.sqrt((n * sum_xx - sum_x * sum_x) * (n * sum_yy - sum_y * sum_y)), 2)
  };
}

// from https://gist.github.com/Tom-Alexander/7166606
export function weightedRegression(logSeconds, watts) {
  const weights = [
    .2,
    .2,
    .3,
    .3,
    .4,
    .4,
    .5,
    .5,
    .6,
    .6,
    .7,
    .7,
    .8,
    .8,
    .9,
    .9,
    1, // 50m
    1,
    1, // 60m
    1,
    1,
    .9,
    .9,
  ];


  let sums = {w: 0, wx: 0, wx2: 0, wy: 0, wxy: 0};

  // compute the weighted averages
  for(let i = 0; i < logSeconds.length; i++){
      sums.w += weights[i];
      sums.wx += logSeconds[i] * weights[i];
      sums.wx2 += logSeconds[i] * logSeconds[i] * weights[i];
      sums.wy += watts[i] * weights[i];
      sums.wxy += logSeconds[i] * watts[i] * weights[i];
  }

  const denominator = sums.w * sums.wx2 - sums.wx * sums.wx;

  let gradient = (sums.w * sums.wxy - sums.wx * sums.wy) / denominator;
  let intercept = (sums.wy * sums.wx2 - sums.wx * sums.wxy) / denominator;

  return {
    slope: gradient,
    intercept,
  };
}
