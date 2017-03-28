<?php
namespace ConvNet;

class SVMLayer extends Layer
{
    public function __construct($opt = [])
    {
        if (empty($opt)) {
            return;
        }

        $this->num_inputs = $opt['in_sx'] * $opt['in_sy'] * $opt['in_depth'];
        $this->out_depth = $this->num_inputs;
        $this->out_sx = 1;
        $this->out_sy = 1;
        $this->layer_type = 'svm';
    }

    public function forward($V, $is_training)
    {
        $this->in_act = $V;
        $this->out_act = $V;
        return $V; // identity function
    }

    public function backward($y = null)
    {
        // compute and accumulate gradient wrt weights and bias of this layer
        $x = $this->in_act;
        $x->dw = array_fill(0, count($x->w), 0); // zero out the gradient of input Vol

        // we're using structured loss here, which means that the score
        // of the ground truth should be higher than the score of any other
        // class, by a margin
        $yscore = $x->w[$y]; // score of ground truth
        $margin = 1.0;
        $loss = 0.0;

        for ($i = 0; $i < $this->out_depth; $i++) {
            if ($y === $i) {
                continue;
            }

            $ydiff = -$yscore + $x->w[$i] + $margin;

            if ($ydiff > 0) {
                // violating dimension, apply loss
                $x->dw[$i] += 1;
                $x->dw[$y] -= 1;
                $loss += $ydiff;
            }
        }

        return $loss;
    }
}
