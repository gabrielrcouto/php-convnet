<?php
namespace ConvNet;

class Net
{
    public $layers;

    public function __construct()
    {
        $this->layers = [];
    }

    public function makeLayers($defs)
    {
        // few checks
        if (count($defs) < 2) {
            throw new \Exception('Error! At least one input layer and one loss layer are required.', 1);
        }

        if ($defs[0]['type'] !== 'input') {
            throw new \Exception('Error! First layer must be the input layer, to declare size of inputs', 1);
        }

        // desugar layer_defs for adding activation, dropout layers etc
        $desugar = function ($defs) {
            $new_defs = [];

            for ($i = 0; $i < count($defs); $i++) {
                $def = $defs[$i];

                if ($def['type'] === 'softmax' || $def['type'] === 'svm') {
                    // add an fc layer here, there is no reason the user should
                    // have to worry about this and we almost always want to
                    $new_defs[] = ['type' => 'fc', 'num_neurons' => $def['num_classes']];
                }

                if ($def['type'] === 'regression') {
                    // add an fc layer here, there is no reason the user should
                    // have to worry about this and we almost always want to
                    $new_defs[] = ['type' => 'fc', 'num_neurons' => $def['num_neurons']];
                }

                if (($def['type'] === 'fc' || $def['type'] === 'conv') && ! array_key_exists('bias_pref', $def)) {
                    $def['bias_pref'] = 0.0;

                    if (array_key_exists('activation', $def) && $def['activation'] === 'relu') {
                        $def['bias_pref'] = 0.1; // relus like a bit of positive bias to get gradients early
                        // otherwise it's technically possible that a relu unit will never turn on (by chance)
                        // and will never get any gradient and never contribute any computation. Dead relu.
                    }
                }

                $new_defs[] = $def;

                if (array_key_exists('activation', $def)) {
                    if ($def['activation'] === 'relu') {
                        $new_defs[] = ['type' => 'relu'];
                    } else if ($def['activation'] === 'sigmoid') {
                        $new_defs[] = ['type' => 'sigmoid'];
                    } else if ($def['activation'] === 'tanh') {
                        $new_defs[] = ['type' => 'tanh'];
                    } else if ($def['activation'] === 'maxout') {
                        // create maxout activation, and pass along group size, if provided
                        $gs = array_key_exists('group_size', $def) ? $def['group_size'] : 2;
                        $new_defs[] = ['type' => 'maxout', 'group_size' => $gs];
                    } else {
                        echo 'ERROR unsupported activation ' + $def['activation'] . PHP_EOL;
                    }
                }

                if (array_key_exists('drop_prob', $def) && $def['type'] !== 'dropout') {
                    $new_defs[] = ['type' => 'dropout', 'drop_prob' => $def['drop_prob']];
                }
            }

            return $new_defs;
        };

        $defs = $desugar($defs);

        // create the layers
        $this->layers = [];

        for ($i = 0; $i < count($defs); $i++) {
            $def = $defs[$i];

            if ($i > 0) {
                $prev = $this->layers[$i - 1];
                $def['in_sx'] = $prev->out_sx;
                $def['in_sy'] = $prev->out_sy;
                $def['in_depth'] = $prev->out_depth;
            }

            switch ($def['type']) {
                case 'fc':
                    $this->layers[] = new FullyConnLayer($def);
                    break;
                case 'lrn':
                    $this->layers[] = new LocalResponseNormalizationLayer($def);
                    break;
                case 'dropout':
                    $this->layers[] = new DropoutLayer($def);
                    break;
                case 'input':
                    $this->layers[] = new InputLayer($def);
                    break;
                case 'softmax':
                    $this->layers[] = new SoftmaxLayer($def);
                    break;
                case 'regression':
                    $this->layers[] = new RegressionLayer($def);
                    break;
                case 'conv':
                    $this->layers[] = new ConvLayer($def);
                    break;
                case 'pool':
                    $this->layers[] = new PoolLayer($def);
                    break;
                case 'relu':
                    $this->layers[] = new ReluLayer($def);
                    break;
                case 'sigmoid':
                    $this->layers[] = new SigmoidLayer($def);
                    break;
                case 'tanh':
                    $this->layers[] = new TanhLayer($def);
                    break;
                case 'maxout':
                    $this->layers[] = new MaxoutLayer($def);
                    break;
                case 'svm':
                    $this->layers[] = new SVMLayer($def);
                    break;
                default:
                    echo 'ERROR: UNRECOGNIZED LAYER TYPE: ' . $def['type'];
            }
        }
    }

    public function forward($V, $is_training = false)
    {
        $act = $this->layers[0]->forward($V, $is_training);

        for ($i = 1; $i < count($this->layers); $i++) {
            $act = $this->layers[$i]->forward($act, $is_training);
        }

        return $act;
    }

    public function getCostLoss($V, $y)
    {
        $this->forward($V, false);
        $N = count($this->layers);
        $loss = $this->layers[$N - 1]->backward($y);

        return $loss;
    }

    public function backward($y)
    {
        $N = count($this->layers);
        $loss = $this->layers[$N - 1]->backward($y); // last layer assumed to be loss layer

        for ($i = $N - 2; $i >= 0; $i--) { // first layer assumed input
            $this->layers[$i]->backward();
        }

        return $loss;
    }

    public function getParamsAndGrads()
    {
        // accumulate parameters and gradients for the entire network
        $response = [];

        for ($i = 0; $i < count($this->layers); $i++) {
            $layer_reponse = $this->layers[$i]->getParamsAndGrads();

            for ($j = 0; $j < count($layer_reponse); $j++) {
                $response[] = &$layer_reponse[$j];
            }
        }

        return $response;
    }

    public function getPrediction()
    {
        // this is a convenience function for returning the argmax
        // prediction, assuming the last layer of the net is a softmax
        $S = $this->layers[count($this->layers) - 1];

        if ($S->layer_type !== 'softmax') {
            throw new \Exception('getPrediction function assumes softmax as last layer of the net!', 1);
        }

        $p = $S->out_act->w;
        $maxv = $p[0];
        $maxi = 0;

        for ($i = 1; $i < count($p); $i++) {
            if ($p[$i] > $maxv) {
                $maxv = $p[$i];
                $maxi = $i;
            }
        }

        return $maxi; // return index of the class with highest class probability
    }

    public function save($file)
    {
        if (file_exists($file)) {
            unlink($file);
        }

        file_put_contents($file, json_encode($this));
    }

    public function load($file)
    {
        if (! file_exists($file)) {
            throw new \Exception('File not found', 1);
        }

        $json = json_decode(file_get_contents($file), true);

        $this->layers = [];

        foreach ($json['layers'] as $key => $layer) {
            switch ($layer['layer_type']) {
                case 'input':
                    $L = new InputLayer();
                    break;
                case 'relu':
                    $L = new ReluLayer();
                    break;
                case 'sigmoid':
                    $L = new SigmoidLayer();
                    break;
                case 'tanh':
                    $L = new TanhLayer();
                    break;
                case 'dropout':
                    $L = new DropoutLayer();
                    break;
                case 'conv':
                    $L = new ConvLayer();
                    break;
                case 'pool':
                    $L = new PoolLayer();
                    break;
                case 'lrn':
                    $L = new LocalResponseNormalizationLayer();
                    break;
                case 'softmax':
                    $L = new SoftmaxLayer();
                    break;
                case 'regression':
                    $L = new RegressionLayer();
                    break;
                case 'fc':
                    $L = new FullyConnLayer();
                    break;
                case 'maxout':
                    $L = new MaxoutLayer();
                    break;
                case 'svm':
                    $L = new SVMLayer();
                    break;
                default:
                    throw new \Exception('Invalid Layer Type ', 1);
            }

            $L->fromJSON($layer);

            $this->layers[] = $L;
        }
    }
}
