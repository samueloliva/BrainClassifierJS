const { CorpusLookup } = require('@nlpjs/utils');
const { TokenizerEn } = require('@nlpjs/lang-en');
const { NeuralNetwork } = require('brain.js');

const defaultNetOptions = {
    log: (str) => console.log(str),
    logPeriod: 10,
    hiddenLayers: [],
    activation: 'leaky-relu',
    learningRate: 0.7,
    momentum: 0.5,
    alpha: 0.08,
    errorThresh: 0.00005,
}

class BrainClassifier {
    constructor(settings, stemmer) {
        this.settings = settings || defaultNetOptions;
        this.tokenizer = new TokenizerEn();
        this.stemmer = stemmer || { tokenizeAndStem: (str) => this.tokenizer.tokenize(str, true) }
    }

    train(corpus) {
        this.lookups = new CorpusLookup(corpus, this.stemmer);
        this.net = new NeuralNetwork();
        this.net.train(this.lookups.trainVectors, this.settings);
    }

    process(utterance) {
        const vector = this.lookups.inputToVector(utterance);
        const output = this.net.run(vector);
        return this.lookups.vectorToClassifications(output);
    }

}

module.exports = BrainClassifier;

/**
 * good morning  -> greet   [1, 1, 0, 0, 0, 0, 0]
 * good night    -> greet   [1, 0, 1, 0, 0, 0, 0]
 * see you later -> bye     [0, 0, 0, 1, 1, 1, 0]
 * good bye      -> bye     [1, 1, 0, 0, 0, 0, 1]
 * good morning night see you later good bye
 * [0, 0, 0, 0, 0, 0, 0]
 * [0.1, 0.3]
 */