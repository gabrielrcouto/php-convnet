<?php
namespace ConvNet;

class InputLayer extends Layer
{
    public function __construct($opt = [])
    {
        if (empty($opt)) {
            return;
        }

        // required: depth
        $this->out_depth = Util::getopt($opt, ['out_depth', 'depth'], 0);

        // optional: default these dimensions to 1
        $this->out_sx = Util::getopt($opt, ['out_sx', 'sx', 'width'], 1);
        $this->out_sy = Util::getopt($opt, ['out_sy', 'sy', 'height'], 1);

        // computed
        $this->layer_type = 'input';
    }

    public function forward($V, $is_training)
    {
        $this->in_act = $V;
        $this->out_act = $V;

        return $this->out_act; // simply identity function for now
    }
}
