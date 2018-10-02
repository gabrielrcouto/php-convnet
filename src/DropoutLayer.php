<?php
namespace ConvNet;

class DropoutLayer extends Layer
{
    public function __construct($opt = [])
    {
        if (empty($opt)) {
            return;
        }

        $this->out_sx = $opt['in_sx'];
        $this->out_sy = $opt['in_sy'];
        $this->out_depth = $opt['in_depth'];
        $this->layer_type = 'dropout';
        $this->drop_prob = array_key_exists('drop_prob', $opt) ? $opt['drop_prob'] : 0.5;
        $this->dropped = array_fill(0, $this->out_sx * $this->out_sy * $this->out_depth, 0);
    }

    public function forward($V, $is_training = false)
    {
        $this->in_act = $V;

        $V2 = $V->clone();
        $N = count($V->w);

        if ($is_training) {
            // do dropout
            for ($i = 0; $i < $N; $i++) {
                if ((rand()/getrandmax()) < $this->drop_prob) {
                    $V2->w[$i] = 0;
                    $this->dropped[$i] = true;
                    // drop!
                } else {
                    $this->dropped[$i] = false;
                }
            }
        } else {
            // scale the activations during prediction
            for ($i = 0; $i < $N; $i++) {
                $V2->w[$i] *= $this->drop_prob;
            }
        }

        $this->out_act = $V2;

        return $this->out_act; // dummy identity function for now
    }

    public function backward($y = null)
    {
        $V = $this->in_act; // we need to set dw of this
        $chain_grad = $this->out_act;
        $N = count($V->w);

        $V->dw = array_fill(0, $N, 0); // zero out gradient wrt data
        for ($i = 0; $i < $N; $i++) {
            if (!($this->dropped[$i])) {
                $V->dw[$i] = $chain_grad->dw[$i]; // copy over the gradient
            }
        }
    }
}
