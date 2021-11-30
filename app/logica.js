let dato = document.getElementsByName('entrada');
let envioDatos = document.querySelector('#envioDatos')

envioDatos.addEventListener('click', (e) => {
    e.preventDefault()
    learnLinear(parseInt(dato[0].value), 10)
})

const learnLinear = async (data) => {
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 1, inputShape: [1] }))

    model.compile({
        loss: 'meanSquaredError',
        optimizer: 'sgd'
    })
    const xs = tf.tensor2d([-1, 0, 1, 2, 3, 4], [6, 1])
    const ys = tf.tensor2d([-3, -1, 1, 3, 5, 7], [6, 1])

    await model.fit(xs, ys, { epochs: 250 })

    console.log(data)


    document.querySelector('#output_field').innerHTML =
        model.predict(tf.tensor2d([data], [1, 1]))
}

const showResults = () => {
    let data = [{
        x: [-1, 0, 1, 2, 3, 4],
        y: [-3, -1, 1, 3, 5, 7],
        type: "scatter"
    }];
    Plotly.newPlot("grafico", data);
}

showResults()



