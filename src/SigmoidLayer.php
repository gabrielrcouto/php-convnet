<?php
namespace ConvNet;

class SigmoidLayer extends Layer
{
    public function __construct($opt = [])
    {
        if (empty($opt)) {
            return;
        }

        $this->out_sx = $opt['in_sx'];
        $this->out_sy = $opt['in_sy'];
        $this->out_depth = $opt['in_depth'];
        $this->layer_type = 'sigmoid';
    }

    public function forward($V, $is_training)
    {
        $this->in_act = $V;
        $V2 = $V->cloneAndZero();
        $N = count($V->w);
        $V2w = &$V2->w;
        $Vw = $V->w;

        for ($i = 0; $i < $N; $i++) {
            $V2w[$i] = 1.0 / (1.0 + exp(-1 * $Vw[$i]));
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
            $v2wi = $V2->w[$i];
            $V->dw[$i] = $v2wi * (1.0 - $v2wi) * $V2->dw[$i];
        }
    }
}
