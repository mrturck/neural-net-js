const nj = require('numjs')

class Helper {
        
    sigmoid_prime(z) {
        let sig = nj.sigmoid(z)
        let ones = nj.ones(z.shape)
        let int = nj.subtract(ones,sig)
        return (sig.multiply(int))
        // derivative of the sigmoid function
    }

    print_details(title="", arr) {
        console.log(`Report: ${title}`.magenta)
        arr.forEach( ar => {
        console.log(`Shape: [${ar.shape}] Dimensions: ${ar.ndim} Length: ${ar.size}`)
        })
    }

    // source: https://bost.ocks.org/mike/shuffle/
    shuffle(array) {
        var m = array.length, t, i;
    
            // While there remain elements to shuffle…
            while (m) {
        
            // Pick a remaining element…
            i = Math.floor(Math.random() * m--);
        
            // And swap it with the current element.
            t = array[m];
            array[m] = array[i];
            array[i] = t;
        }
    
        return array;
    }
 
    zip(a,b) {
        let c = a.map(function(e, i) {
            return [e, b[i]];
        });
        return c
    }

    // source: https://gist.github.com/engelen/fbce4476c9e68c52ff7e5c2da5c24a28
    argMax(array) {
        // this is just a dummy until i get better data
        return Math.round(array.tolist())
        // true argmax below
        // return array.map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
    }

    getCol(array,col) {
        let c = []
        for (let i in array) {c.push([array[i][col]])}
        return c
    }
}
module.exports = {
    Helper
};
