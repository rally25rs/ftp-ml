import * as tf from '@tensorflow/tfjs';
import * as path from 'node:path';
import * as url from 'node:url';

const dirname = url.fileURLToPath(new URL('.', import.meta.url));
const filename = `file://${dirname}/event.csv`;

const config1 = {
  optimizer: 'adam',
  loss: tf.losses.absoluteDifference,
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

// mse 6.6
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
    tf.layers.dense({
      inputShape: [numOfFeatures],
      units: Math.round(numOfFeatures * 2)
    }),
    tf.layers.dense({
      units: 1
    }),
  ]),
};

const config = config7;

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

const numOfFeatures = (await csvDataset.columnNames()).length - 1;

// Prepare the Dataset for training.
const flattenedDataset = csvDataset.map(({xs, ys}) =>
  {
    // Convert xs(features) and ys(labels) from object form (keyed by column name) to array form.
    return {xs:Object.values(xs), ys:Object.values(ys)};
  }).batch(10);

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
    {
      onEpochEnd: async (epoch, logs) => {
        if(epoch % 10 === 0) {
          console.log(epoch + ':' + JSON.stringify(logs));
        }
      },
    },
    // tf.callbacks.earlyStopping({
    //   monitor: 'loss',
    //   patience: 5,
    // })
  ],

  // callbacks: {
  //   onEpochEnd: async (epoch, logs) => {
  //     // if(epoch === epochs - 1) {
  //       console.log(epoch + ':' + logs.loss);
  //     // }
  //   },
  // }
});

// 2582903,365,357,349,340,333,322,316,313,311,308,302,301,298,296,295,294,293,293,293,294,294,294,292,290,290,289,288,287,284,284,258,259,259,205,202,201,201,201,279
model.predict(tf.tensor2d([[365,357,349,340,333,322,316,313,311,308,302,301,298,296,295,294,293,293,293,294,294,294,292,290,290,289,288,287,284,284,258,259,259,205,202,201,201,201]], [1, 38])).print();
console.log('Expected Actual: 279');
model.predict(tf.tensor2d([[468,471,431,396,379,373,371,368,369,364,364,358,359,359,358,358,347,338,332,327,324,322,323,321,319,317,312,311,309,304,296,289,287,287,281,282,281,0]], [1, 38])).print();
console.log('Expected Actual: 293');
