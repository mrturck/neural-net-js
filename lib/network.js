// NOTE: Unable to perform vector based training due to limitation of numjs library

const nj = require('numjs')
const h = require('./helper')
const helper = new h.Helper()

class Network {

    constructor (sizes) {
      this.sizes = sizes
      this.num_layers = sizes.length
      this.biases = this.gen_biases(sizes)
      this.weights = this.gen_weights(sizes)
    }

    gen_biases(sizes) {
      // returns array of random biases between 0 and 1 
      // excluding first layer (index 0)
      const b = []
      for (let i=1;i<sizes.length;i++) {
           b.push(nj.random([sizes[i],1]))
           }
        return b
    }

    gen_weights(sizes) {
        // returns array of weights
       const w = []
       const mapped = sizes.slice(0,-1).map( (e,i) => {
           return [sizes.slice(1)[i],e]
       })
       mapped.forEach( (item) => {
           w.push(nj.random([item[0],item[1]]));
       })
        return w
    }

    predict(array) {
        // input a tuple of n-length (ex: [2,3,1])
        // output result of network
        let res = []
        array.forEach( i => {
            res.push(this.feedforward(i).tolist())
        })
        return res
    }

    feedforward(input) {
        let a = nj.array([input]).T
        let w = this.weights
        let b = this.biases
        for (let i=0;i<w.length;i++) {
            a = nj.sigmoid(nj.dot(w[i],a).add(b[i]))
        }
        return a
    }

    train(training_data, epochs, batch_size, learning_rate, test_data=null, log=true) {
            // SGD - stochastic gradient descent
        let n_test = 0  
        if (test_data) {
            n_test = test_data.length
        }

        let n = training_data.length
        let j = 0

        while (j < epochs) {
            // shuffle training data and create mini-batches
            let minibatches = []
            let data = helper.shuffle(training_data)
            this.create_batches(data,minibatches,batch_size,n)
            // perform SGD on each batch
            minibatches.forEach( batch => {
                this.update_batch(batch,learning_rate)
            })
            // evaluate results at the end of the epoch
            if (log)
           { if (test_data) {
            //    console.log(test_data)
                let res = this.evaluate(test_data)
                if ( res > n_test * .5) {
                  console.log(`Epoch ${j}: ${res}/${n_test}: `.green )
                }
                else {
                  console.log(`Epoch ${j}: ${res}/${n_test}: `.red )
                }
            }
            else {
              console.log(`Epoch ${j+1} complete`)
            }}
            j += 1
        }
        return 1
    }

    update_batch(batch, learning_rate) {
        //input: list of tuples -- verified
        //output: updates weights and biases
        // updates network weights and biases 
        // applies gradient descent by backpropagation algorithm on the given batch
        let nabla_b = []
        let nabla_w = []

        for (let i=0;i<this.biases.length;i++) { nabla_b.push(nj.zeros(this.biases[i].shape)) }
        for (let i=0;i<this.weights.length;i++) {nabla_w.push(nj.zeros(this.weights[i].shape))}

        //FIXME: Implement vector backprop
        // 1. Pull out the
        for (var [x,y] of batch) { 
            let [delta_nabla_b, delta_nabla_w] = this.backprop(x,y) // return backprop tuple
            for (let i=0;i<helper.zip(nabla_b,delta_nabla_b).length;i++) {
                nabla_b[i] = nabla_b[i].add(delta_nabla_b[i])
            }
            
            for (let i=0;i<helper.zip(nabla_w,delta_nabla_w).length;i++) {
                nabla_w[i] = nabla_w[i].add(delta_nabla_w[i])
            }
        }

        for (let i=0;i<helper.zip(this.weights,nabla_w).length;i++) {
            this.weights[i] = (this.weights[i]
                .subtract(
                    nabla_w[i].multiply((learning_rate/(batch.length)))
                )
            )
        }

        for (let i=0;i<helper.zip(this.biases,nabla_b).length;i++) {
            this.biases[i] = (this.biases[i].subtract(nabla_b[i].multiply((learning_rate/(batch.length)))))
        }
    }

    backprop(x,y) {
        // backprop updates weights and biases for a single x,y pair that is passed
        // the following nabla's are layer-by-layer lists of arrays
        let nabla_w = []
        let nabla_b = []
        
        for (let i=0;i<this.biases.length;i++) {nabla_b.push(nj.zeros(this.biases[i].shape))}
        for (let i=0;i<this.weights.length;i++) {nabla_w.push(nj.zeros(this.weights[i].shape))}

        // feedforward
        var activation = nj.array([x]).T
        var  activations = [activation] // stores activations each layer
        let zs = [] // stores z, or weighted inputs, each layer
        for (var [b,w] of helper.zip(this.biases,this.weights)) {
            let z = nj.dot(w,activation).add(b)
            zs.push(z)
            activation = nj.sigmoid(z)
            activations.push(activation)
        }


        // backward pass
        let delta = this.cost_derivative(activations.slice(-1)[0],[y]).multiply(helper.sigmoid_prime(zs.slice(-1)[0]))
        nabla_b[(nabla_b.length-1)] = delta
        nabla_w[(nabla_w.length-1)] = nj.dot(delta, activations.slice(-2)[0].T)

        for (let i=2;i<this.num_layers;i++) {
            let z = zs.slice(-i)[0]
            let sp = helper.sigmoid_prime(z)

            delta = nj.dot(this.weights.slice(-i+1)[0].T,delta).multiply(sp)
            
            nabla_b[(nabla_b.length-i)] = delta
            nabla_w[(nabla_w.length-i)] = nj.dot(delta, activations.slice(-i-1)[0].T)
        }
        return ([nabla_b,nabla_w])

    }

    create_batches(initial_list, final_list, batch_size, total) {
        for (let k=0;k<total;k+=batch_size) {
            let batch = initial_list.slice(k,k+batch_size)
            if (batch.length > 0) {
                final_list.push(batch)
                }
            }
        return final_list
    }

    evaluate(test_data) {
        // evaluates model against test_data, returns accuracy
        // neural network output is assumed to be index of whichever
        // neuron in final layer has highest activation
        let test_results = []
        for (let [x,y] of test_data) {
          test_results.push([argMax(this.feedforward(x)),y])
        }
        let sum = 0
        for (let [x,y] of test_results) {
          if ( x*1 == y*1) {sum++}
        }
        console.log(test_results)
        return sum
    }

    cost_derivative(output_activations, y) {
        // returns partial dC_x/da
        return (nj.subtract(output_activations,nj.array([y])))
    }
}

module.exports = {
    Network
};
