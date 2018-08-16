const net = require('../lib/network');
const nj = require('numjs');
const colors = require('colors');
const h = require('../lib/helper')
const helper = new h.Helper()

console.log('Begin testing...'.blue)
let score = 0
let count = 0

//helper function
function assert(input_name, input, output) {
    try {
        input;
        if (input == output) {
            console.log(`${input_name} : passed`); score+=1;count+=1;
        }
        else {
            console.log(`${input_name} : failed`);count+=1;
            console.log(`${input}`.red)
        }
    
    }
    catch(error) {
        console.log(error)
    }
    
}

// Test suite

    // generate training data 
    // TODO: load training data in future
    let o = Array.from(new Array(1000),(val,index)=>(.25));
    let i = Array.from(new Array(1000),(val,index)=>([index/100]))
    let input = helper.zip(i,o)
    // console.log(input)

    // class created successfully -- verified by num_layers
    const network = new net.Network([1,6,1])

    assert('network creation',network.num_layers,3)

    // class creates biases successfully 
    assert('biases initialized',network.biases[0].shape.toString(),[ 6, 1 ].toString())

    // class creates weights successfully
    assert('weights initialized',network.weights[0].shape.toString(), [ 6, 1 ].toString())

    // test that feedforward works correctly
    console.log(`Initial guess: [${network.predict([[1],[2]])}] Target:${input[0][1]}`.blue)

    // console.log(`Begin: [${network.predict([[1,20],[3,-4]])}] Target:${input[0][1]}`.blue)
    network.train(input ,30 ,2 ,3 ,'',false)
    // console.log(`Final: [${network.predict([[1,20],[3,-4]])}] Target:${input[0][1]}`.blue)
    console.log(`Final guess: [${network.predict([[1],[20]])}] Target:${input[0][1]}`.blue)

    // test that SGD runs
    assert('SGD Runs',network.train(input,3,10,4,input,false),1)
    
    // check that cost_derivative works
    assert('cost_derivative',network.cost_derivative([[1,3]],[1,-2]).tolist(),'0,5')

    // check that sigmoid_prime works
    a = nj.array([[ 0.18982, 0.43081],[ 0.80713, 0.47704]])
    assert('sigmoid_prime',helper.sigmoid_prime(a),nj.array([[ 0.24776, 0.23875],[ 0.21333,  0.2363]]).toString())

    // check argMax (temporary that it rounds correctly)
    assert('argMax',helper.argMax(nj.array([3.44156])),3)
    //TODO: Figure out why SGD doesn't update weights and biases correctly and test here

// Output test results
console.log(`${score}/${count} tests successful`.green)
