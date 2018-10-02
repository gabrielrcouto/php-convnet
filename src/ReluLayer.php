<?php
namespace ConvNet;

class ReluLayer extends Layer
{
    public function __construct($opt = [])
    {
        if (empty($opt)) {
            return;
        }

        $this->out_sx = $opt['in_sx'];
        $this->out_sy = $opt['in_sy'];
        $this->out_depth = $opt['in_depth'];
        $this->layer_type = 'relu';
    }

    public function forward($V, $is_training)
    {
        $this->in_act = $V;
        $V2 = $V->clone();
        $N = count($V->w);
        $V2w = &$V2->w;

        for ($i = 0; $i < $N; $i++) {
            if ($V2w[$i] < 0) {
                $V2w[$i] = 0; // threshold at 0
            }
        }

        $this->out_act = $V2;

        return $this->out_act;
    }

    public function backward($y = null)
    {
        $V = $this->in_act; // we need to set dw of this
        $V2 = $this->out_act;
        $N = count($V->w);
        $V->dw = array_fill(0, $N, 0); // zero out gradient wrt data

        for ($i = 0; $i < $N; $i++) {
            if ($V2->w[$i] <= 0) {
                $V->dw[$i] = 0; // threshold
            } else {
                $V->dw[$i] = $V2->dw[$i];
            }
        }
    }
}
