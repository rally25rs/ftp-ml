import * as tf from '@tensorflow/tfjs-node';
import * as path from 'node:path';
import * as url from 'node:url';
import { linearRegression } from './regressions.mjs'

const dirname = url.fileURLToPath(new URL('.', import.meta.url));
const filename = `file://${dirname}event.csv`;

// usage: node index.mjs {model}
const saveModelToFile = process.argv[2] ? `file://${dirname}${process.argv[2]}` : undefined;

// mae 0.14
const config1 = {
  optimizer: 'adam',
  loss: tf.losses.absoluteDifference,
  metrics: [
    tf.metrics.meanAbsoluteError,
  ],
  epochs: 300,
  batch: 32,
  layers: (numOfFeatures => [
    tf.layers.dense({
      inputShape: [numOfFeatures],
      units: Math.abs(numOfFeatures)
    }),
    tf.layers.dense({
      units: 1
    }),
  ]),
};

// mae .196 mse .12
const config1b = {
  optimizer: 'adam',
  loss: tf.losses.meanSquaredError,
  metrics: [
    tf.metrics.meanAbsoluteError,
    tf.metrics.meanSquaredError,
  ],
  epochs: 700, //360,
  layers: (numOfFeatures => [
    tf.layers.dense({
      inputShape: [numOfFeatures],
      units: Math.abs(numOfFeatures)
    }),
    tf.layers.dense({
      units: 1
    }),
  ]),
};

const config1c = {
  optimizer: 'adam',
  loss: tf.losses.meanSquaredError,
  metrics: [
    tf.metrics.meanAbsoluteError,
    // tf.metrics.meanSquaredError,
  ],
  epochs: 400,
  layers: (numOfFeatures => [
    tf.layers.dense({
      inputShape: [numOfFeatures],
      units: Math.round(numOfFeatures / 2)
    }),
    tf.layers.dense({
      units: 1
    }),
  ]),
};

// mse 6.8
const config2 = {
  optimizer: 'adam',
  loss: tf.losses.meanSquaredError,
  metrics: [
    tf.metrics.meanAbsoluteError,
  ],
  epochs: 300,
  layers: (numOfFeatures => [
    tf.layers.dense({
      inputShape: [numOfFeatures],
      units: Math.abs(numOfFeatures / 2)
    }),
    tf.layers.dense({
      units: 1
    }),
  ]),
};

// mse 6.75, 7.7
const config3 = {
  optimizer: 'adam',
  loss: tf.losses.meanSquaredError,
  metrics: [
    tf.metrics.meanAbsoluteError,
  ],
  epochs: 100,
  layers: (numOfFeatures => [
    tf.layers.dense({
      inputShape: [numOfFeatures],
      units: 1
    }),
  ]),
};

// mse 6.8
const config4 = {
  optimizer: tf.OptimizerConstructors.adamax(),
  loss: tf.losses.meanSquaredError,
  metrics: [
    tf.metrics.meanAbsoluteError,
  ],
  epochs: 200,
  layers: (numOfFeatures => [
    tf.layers.dense({
      inputShape: [numOfFeatures],
      units: Math.round(numOfFeatures * 0.5)
    }),
    tf.layers.dense({
      units: 1
    }),
  ]),
};

// mse 6.5
const config5 = {
  optimizer: tf.OptimizerConstructors.adadelta(.0000002),
  loss: tf.losses.meanSquaredError,
  metrics: [
    tf.metrics.meanAbsoluteError,
  ],
  epochs: 100,
  layers: (numOfFeatures => [
    tf.layers.dense({
      inputShape: [numOfFeatures],
      units: Math.round(numOfFeatures * 0.5)
    }),
    tf.layers.dense({
      units: 1
    }),
  ]),
};

// 7.8
const config6 = {
  optimizer: tf.OptimizerConstructors.sgd(.0000002),
  loss: tf.losses.meanSquaredError,
  metrics: [
    tf.metrics.meanAbsoluteError,
  ],
  epochs: 100,
  layers: (numOfFeatures => [
    tf.layers.dense({
      inputShape: [numOfFeatures],
      units: Math.round(numOfFeatures * 0.5)
    }),
    tf.layers.dense({
      units: 1
    }),
  ]),
};

// 7.2
const config7 = {
  optimizer: 'adam',
  loss: tf.losses.meanSquaredError,
  metrics: [
    tf.metrics.meanAbsoluteError,
  ],
  epochs: 100,
  layers: (numOfFeatures => [
    tf.layers.conv1d({
      inputShape: [numOfFeatures],
      units: Math.round(numOfFeatures * 2),
      kernelSize: 2,
      filters: 1,
    }),
    tf.layers.dense({
      units: 1
    }),
  ]),
};

const config = config1;

const powerColumnConfig = { isLabel: false, dtype: 'int32' };
const csvDataset = tf.data.csv(
  filename, {
    hasHeader: false,
    columnNames: [
      'athleteId',
      '120',
      '150',
      '180',
      '210',
      '240',
      '270',
      '300',
      '330',
      '360',
      '390',
      '420',
      '480',
      '510',
      '540',
      '570',
      '600',
      '660',
      '720',
      '780',
      '840',
      '900',
      '960',
      '1020',
      '1080',
      '1140',
      '1200',
      '1500',
      '1800',
      '2100',
      '2400',
      '2700',
      '3000',
      '3300',
      '3600',
      '4200',
      '4800',
      '5400',
      '6000',
      'zftp'
    ],
    columnConfigs: {
      '120': powerColumnConfig,
      '150': powerColumnConfig,
      '180': powerColumnConfig,
      '210': powerColumnConfig,
      '240': powerColumnConfig,
      '270': powerColumnConfig,
      '300': powerColumnConfig,
      '330': powerColumnConfig,
      '360': powerColumnConfig,
      '390': powerColumnConfig,
      '420': powerColumnConfig,
      '480': powerColumnConfig,
      '510': powerColumnConfig,
      '540': powerColumnConfig,
      '570': powerColumnConfig,
      '600': powerColumnConfig,
      '660': powerColumnConfig,
      '720': powerColumnConfig,
      '780': powerColumnConfig,
      '840': powerColumnConfig,
      '900': powerColumnConfig,
      '960': powerColumnConfig,
      '1020': powerColumnConfig,
      '1080': powerColumnConfig,
      '1140': powerColumnConfig,
      '1200': powerColumnConfig,
      '1500': powerColumnConfig,
      '1800': powerColumnConfig,
      '2100': powerColumnConfig,
      '2400': powerColumnConfig,
      '2700': powerColumnConfig,
      '3000': powerColumnConfig,
      '3300': powerColumnConfig,
      '3600': powerColumnConfig,
      '4200': powerColumnConfig,
      '4800': powerColumnConfig,
      '5400': powerColumnConfig,
      '6000': powerColumnConfig,
      zftp: { isLabel: true, dtype: 'int32' }
    },
    configuredColumnsOnly: true,
  });

const columnNames = await csvDataset.columnNames();
const numOfFeatures = columnNames.length - 1;
const logTimes = columnNames.map(n => parseInt(n)).filter(n => !isNaN(n)).map(n => Math.log(n));
const backfillStart = logTimes.indexOf(Math.log(600)); // use linear regression from 10m+ to backfill zeros

class StopEarly extends tf.Callback {
  constructor() {
    super();
  }

  async onEpochEnd(epoch, logs) {
    if(logs.loss < .14) {
      this.model.stopTraining = true;
    }
  }
}

function replaceZeros(powers) {
  const regression = linearRegression(logTimes.slice(backfillStart), powers.slice(backfillStart));
  for(let i = powers.length - 1; i >= 0; i--) {
    if(powers[i] === 0) {
      powers[i] = regression.slope * logTimes[i] + regression.intercept;
    } else {
      break;
    }
  }
}

// Prepare the Dataset for training.
let flattenedDataset = csvDataset.map(({xs, ys}) =>
  {
    // Convert xs(features) and ys(labels) from object form (keyed by column name) to array form.
    const featureWkgs = Object.values(xs);
    replaceZeros(featureWkgs);
    return {xs:featureWkgs, ys:Object.values(ys)};
  }).batch(config.batch || 32);

// Define the model.
const model = tf.sequential();
config.layers(numOfFeatures).forEach(layer => model.add(layer));

model.compile({
  optimizer: config.optimizer,
  loss: config.loss,
  metrics: config.metrics,
});

model.summary();

// Fit the model using the prepared Dataset
await model.fitDataset(flattenedDataset, {
  epochs: config.epochs,
  callbacks: [
    new StopEarly(),
    // {
    //   onEpochEnd: async (epoch, logs) => {
    //     if(epoch % 10 === 0) {
    //       console.log(epoch + ':' + JSON.stringify(logs));
    //     }
    //   },
    // },
    // tf.callbacks.earlyStopping({
    //   monitor: 'loss',
    //   patience: 5,
    // })
  ],
});

if (saveModelToFile) {
  await model.save(saveModelToFile);
  console.log(`Saved model to ${saveModelToFile}`);
}

// 2582903,365,357,349,340,333,322,316,313,311,308,302,301,298,296,295,294,293,293,293,294,294,294,292,290,290,289,288,287,284,284,258,259,259,205,202,201,201,201,279
model.predict(tf.tensor2d([[365,357,349,340,333,322,316,313,311,308,302,301,298,296,295,294,293,293,293,294,294,294,292,290,290,289,288,287,284,284,258,259,259,205,202,201,201,201]], [1, 38])).print();
console.log('Expected Actual: 279');
model.predict(tf.tensor2d([[468,471,431,396,379,373,371,368,369,364,364,358,359,359,358,358,347,338,332,327,324,322,323,321,319,317,312,311,309,304,296,289,287,287,281,282,281,0]], [1, 38])).print();
console.log('Expected Actual: 293');
