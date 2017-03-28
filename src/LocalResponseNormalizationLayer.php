<?php
namespace ConvNet;

class LocalResponseNormalizationLayer extends Layer
{
    public function __construct($opt = [])
    {
        if (empty($opt)) {
            return;
        }

        // required
        $this->k = $opt['k'];
        $this->n = $opt['n'];
        $this->alpha = $opt['alpha'];
        $this->beta = $opt['beta'];

        // computed
        $this->out_sx = $opt['in_sx'];
        $this->out_sy = $opt['in_sy'];
        $this->out_depth = $opt['in_depth'];
        $this->layer_type = 'lrn';

        // checks
        if ($this->n % 2 === 0) {
            echo 'WARNING n should be odd for LRN layer' . PHP_EOL;
        }
    }

    public function forward($V, $is_training)
    {
        $this->in_act = $V;

        $A = $V->cloneAndZero();
        $this->S_cache_ = $V.cloneAndZero();
        $n2 = floor($this->n / 2);

        for ($x = 0; $x < $V->sx; $x++) {
            for ($y = 0; $y < $V->sy; $y++) {
                for ($i = 0; $i < $V->depth; $i++) {
                    $ai = $V->get($x, $y, $i);

                    // normalize in a window of size n
                    $den = 0.0;

                    for ($j = max(0, $i - $n2); $j <= min($i + $n2, $V->depth - 1); $j++) {
                        $aa = $V->get($x, $y, $j);
                        $den += $aa * $aa;
                    }

                    $den *= $this->alpha / $this->n;
                    $den += $this->k;
                    $this->S_cache_.set($x, $y, $i, $den); // will be useful for backprop
                    $den = pow($den, $this->beta);
                    $A->set($x, $y, $i, $ai / $den);
                }
            }
        }

        $this->out_act = $A;

        return $this->out_act; // dummy identity function for now
    }

    public function backward($y = null)
    {
        // evaluate gradient wrt data
        $V = $this->in_act; // we need to set dw of this
        $V->dw = array_fill(0, count($V->w), 0); // zero out gradient wrt data
        $A = $this->out_act; // computed in forward pass

        $n2 = floor($this->n / 2);

        for ($x = 0; $x < $V->sx; $x++) {
            for ($y = 0; $y < $V->sy; $y++) {
                for ($i = 0; $i < $V->depth; $i++) {
                    $chain_grad = $this->out_act->get_grad($x, $y, $i);
                    $S = $this->S_cache_.get($x, $y, $i);
                    $SB = pow($S, $this->beta);
                    $SB2 = $SB * $SB;

                    // normalize in a window of size n
                    for ($j = max(0, $i - $n2); $j <= min($i + $n2, $V->depth - 1); $j++) {
                        $aj = $V->get($x, $y, $j);
                        $g = -$aj * $this->beta * pow($S, $this->beta - 1) * $this->alpha / $this->n * 2 * $aj;

                        if ($j === $i) {
                            $g += $SB;
                        }

                        $g /= $SB2;
                        $g *= $chain_grad;
                        $V->add_grad($x, $y, $j, $g);
                    }
                }
            }
        }
    }
}
