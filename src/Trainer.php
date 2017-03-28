<?php
namespace ConvNet;

class Trainer
{
    public $net;
    public $options;
    public $learning_rate;
    public $l1_decay;
    public $l2_decay;
    public $batch_size;
    public $method;
    public $momentum;
    public $ro;
    public $eps;
    public $beta1;
    public $beta2;
    public $k;
    public $gsum;
    public $xsum;
    public $regression;

    public function __construct($net, $options = [])
    {
        $this->net = $net;

        $this->learning_rate = array_key_exists('learning_rate', $options) ? $options['learning_rate'] : 0.01;
        $this->l1_decay = array_key_exists('l1_decay', $options) ? $options['l1_decay'] : 0.0;
        $this->l2_decay = array_key_exists('l2_decay', $options) ? $options['l2_decay'] : 0.0;
        $this->batch_size = array_key_exists('batch_size', $options) ? $options['batch_size'] : 1;
        $this->method = array_key_exists('method', $options) ? $options['method'] : 'sgd'; // sgd/adam/adagrad/adadelta/windowgrad/netsterov

        $this->momentum = array_key_exists('momentum', $options) ? $options['momentum'] : 0.9;
        $this->ro = array_key_exists('ro', $options) ? $options['ro'] : 0.95; // used in adadelta
        $this->eps = array_key_exists('eps', $options) ? $options['eps'] : 1e-8; // used in adam or adadelta
        $this->beta1 = array_key_exists('beta1', $options) ? $options['beta1'] : 0.9; // used in adam
        $this->beta2 = array_key_exists('beta2', $options) ? $options['beta2'] : 0.999; // used in adam

        $this->k = 0; // iteration counter
        $this->gsum = []; // last iteration gradients (used for momentum calculations)
        $this->xsum = []; // used in adam or adadelta

        // check if regression is expected
        if ($this->net->layers[count($this->net->layers) - 1]->layer_type === 'regression') {
            $this->regression = true;
        } else {
            $this->regression = false;
        }
    }

    public function train($x, $y)
    {
        $start = microtime(true);
        $this->net->forward($x, true); // also set the flag that lets the net know we're just training
        $end = microtime(true);
        $fwd_time = $end - $start;

        $start = microtime(true);
        $cost_loss = $this->net->backward($y);
        $l2_decay_loss = 0.0;
        $l1_decay_loss = 0.0;
        $end = microtime(true);
        $bwd_time = $end - $start;

        if ($this->regression && ! is_array($y)) {
            echo 'Warning: a regression net requires an array as training output vector.' . PHP_EOL;
        }

        $this->k++;

        if ($this->k % $this->batch_size === 0) {
            $pglist = $this->net->getParamsAndGrads();
            // initialize lists for accumulators. Will only be done once on first iteration
            if (count($this->gsum) === 0 && ($this->method !== 'sgd' || $this->momentum > 0.0)) {
                // only vanilla sgd doesnt need either lists
                // momentum needs gsum
                // adagrad needs gsum
                // adam and adadelta needs gsum and xsum
                for ($i = 0; $i < count($pglist); $i++) {
                    $this->gsum[] = array_fill(0, count($pglist[$i]['params']), 0);

                    if ($this->method === 'adam' || $this->method === 'adadelta') {
                        $this->xsum[] = array_fill(0, count($pglist[$i]['params']), 0);
                    } else {
                        $this->xsum[] = []; // conserve memory
                    }
                }
            }

            // perform an update for all sets of weights
            for ($i = 0; $i < count($pglist); $i++) {
                $pg = &$pglist[$i]; // param, gradient, other options in future (custom learning rate etc)
                $p = &$pg['params'];
                $g = &$pg['grads'];

                // learning rate for some parameters.
                $l2_decay_mul = array_key_exists('l2_decay_mul', $pg) ? $pg['l2_decay_mul'] : 1.0;
                $l1_decay_mul = array_key_exists('l1_decay_mul', $pg) ? $pg['l1_decay_mul'] : 1.0;
                $l2_decay = $this->l2_decay * $l2_decay_mul;
                $l1_decay = $this->l1_decay * $l1_decay_mul;

                $plen = count($p);

                for ($j = 0; $j < $plen; $j++) {
                    $l2_decay_loss += $l2_decay * $p[$j] * $p[$j] / 2; // accumulate weight decay loss
                    $l1_decay_loss += $l1_decay * abs($p[$j]);
                    $l1grad = $l1_decay * ($p[$j] > 0 ? 1 : -1);
                    $l2grad = $l2_decay * ($p[$j]);

                    $gij = ($l2grad + $l1grad + $g[$j]) / $this->batch_size; // raw batch gradient

                    $gsumi = $this->gsum[$i];
                    $xsumi = $this->xsum[$i];

                    if ($this->method === 'adam') {
                        // adam update
                        $gsumi[$j] = $gsumi[$j] * $this->beta1 + (1 - $this->beta1) * $gij; // update biased first moment estimate
                        $xsumi[$j] = $xsumi[$j] * $this->beta2 + (1 - $this->beta2) * $gij * $gij; // update biased second moment estimate
                        $biasCorr1 = $gsumi[$j] * (1 - pow($this->beta1, $this->k)); // correct bias first moment estimate
                        $biasCorr2 = $xsumi[$j] * (1 - pow($this->beta2, $this->k)); // correct bias second moment estimate
                        $dx =  - $this->learning_rate * $biasCorr1 / (sqrt($biasCorr2) + $this->eps);
                        $p[$j] += $dx;
                    } else if ($this->method === 'adagrad') {
                        // adagrad update
                        $gsumi[$j] = $gsumi[j] + $gij * $gij;
                        $dx = - $this->learning_rate / sqrt($gsumi[$j] + $this->eps) * $gij;
                        $p[$j] += $dx;
                    } else if ($this->method === 'windowgrad') {
                        // this is adagrad but with a moving window weighted average
                        // so the gradient is not accumulated over the entire history of the run.
                        // it's also referred to as Idea #1 in Zeiler paper on Adadelta. Seems reasonable to me!
                        $gsumi[$j] = $this->ro * $gsumi[$j] + (1 - $this->ro) * $gij * $gij;
                        $dx = - $this->learning_rate / sqrt($gsumi[$j] + $this->eps) * $gij; // eps added for better conditioning
                        $p[$j] += $dx;
                    } else if ($this->method === 'adadelta') {
                        $gsumi[$j] = $this->ro * $gsumi[$j] + (1 - $this->ro) * $gij * $gij;
                        $dx = - sqrt(($xsumi[$j] + $this->eps) / ($gsumi[$j] + $this->eps)) * $gij;
                        $xsumi[$j] = $this->ro * $xsumi[$j] + (1 - $this->ro) * $dx * $dx; // yes, xsum lags behind gsum by 1.
                        $p[$j] += $dx;
                    } else if ($this->method === 'nesterov') {
                        $dx = $gsumi[$j];
                        $gsumi[$j] = $gsumi[$j] * $this->momentum + $this->learning_rate * $gij;
                        $dx = $this->momentum * $dx - (1.0 + $this->momentum) * $gsumi[$j];
                        $p[$j] += $dx;
                    } else {
                        // assume SGD
                        if ($this->momentum > 0.0) {
                            // momentum update
                            $dx = $this->momentum * $gsumi[$j] - $this->learning_rate * $gij; // step
                            $gsumi[$j] = $dx; // back this up for next iteration of momentum
                            $p[$j] += $dx; // apply corrected gradient
                        } else {
                            // vanilla sgd
                            $p[$j] += $this->learning_rate * $gij * (-1);
                        }
                    }

                    $g[$j] = 0.0; // zero out gradient so that we can begin accumulating anew
                }
            }
        }

        // appending softmax_loss for backwards compatibility, but from now on we will always use cost_loss
        // in future, TODO: have to completely redo the way loss is done around the network as currently
        // loss is a bit of a hack. Ideally, user should specify arbitrary number of loss functions on any layer
        // and it should all be computed correctly and automatically.
        return [
            'fwd_time' => $fwd_time,
            'bwd_time' => $bwd_time,
            'l2_decay_loss' => $l2_decay_loss,
            'l1_decay_loss' => $l1_decay_loss,
            'cost_loss' => $cost_loss,
            'softmax_loss' => $cost_loss,
            'loss' => $cost_loss + $l1_decay_loss + $l2_decay_loss
        ];
    }
}
