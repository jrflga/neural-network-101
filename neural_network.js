var synaptic = require("synaptic");

var inputLayer = new synaptic.Layer(2);
var hiddenLayer1 = new synaptic.Layer(4);
var hiddenLayer2 = new synaptic.Layer(4);
var outputLayer = new synaptic.Layer(1);

inputLayer.project(hiddenLayer1);
hiddenLayer1.project(hiddenLayer2);
hiddenLayer2.project(outputLayer);

var myNetwork = new synaptic.Network({
    input: inputLayer,
    hidden: [
        hiddenLayer1,
        hiddenLayer2
    ],
    output: outputLayer
});

var trainer = new synaptic.Trainer(myNetwork);

var trainingSet = [
    {
        input: [0, 0],
        output: [0]
    },
    {
        input: [0, 1],
        output: [1]
    },
    {
        input: [1, 0],
        output: [1]
    },
    {
        input: [1, 1],
        output: [0]
    }
];

trainer.train(
    trainingSet,
    {
        rate: .3, // Learning rate
        iterations: 1000000, // Maximum number of Iterations
        error: .0000005, // Minimum error
        shuffle: true, // Shuffled after every iter. Good for LSTM's (order not meaningful).
        cost: synaptic.Trainer.cost.MSE, // Self-explained
        log: 1 // console.log the error and iter
    }
);

console.log("[0, 0] => [" + myNetwork.activate([0, 0]) + "]");
console.log("[0, 1] => [" + myNetwork.activate([0, 1]) + "]");
console.log("[1, 0] => [" + myNetwork.activate([1, 0]) + "]");
console.log("[1, 1] => [" + myNetwork.activate([1, 1]) + "]");