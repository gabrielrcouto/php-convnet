<?php
namespace ConvNet;

use ComputerVision\ImageReaders\Pgm;

class Vol
{
    public $sx;
    public $sy;
    public $depth;
    public $w;
    public $dw;

    public function __construct($sx, $sy = null, $depth = null, $c = null)
    {
        if (is_array($sx)) {
            // we were given a list in sx, assume 1D volume and fill it up
            $this->sx = 1;
            $this->sy = 1;
            $this->depth = count($sx);
            // we have to do the following copy because we want to use
            // fast typed arrays, not an ordinary javascript array
            $this->w = array_fill(0, $this->depth, 0);
            $this->dw = array_fill(0, $this->depth, 0);

            for ($i = 0; $i < $this->depth; $i++) {
                $this->w[$i] = $sx[$i];
            }
        } else {
            // we were given dimensions of the vol
            $this->sx = $sx;
            $this->sy = $sy;
            $this->depth = $depth;
            $n = $sx * $sy * $depth;

            $this->w = array_fill(0, $n, 0);
            $this->dw = array_fill(0, $n, 0);

		    if ($c === null) {
                // weight normalization is done to equalize the output
                // variance of every neuron, otherwise neurons with a lot
                // of incoming connections have outputs of larger variance
                $scale = sqrt(1.0 / ($sx * $sy * $depth));

                for ($i = 0; $i < $n; $i++) {
                    $this->w[$i] = (rand()*$scale)/getrandmax();
                }
                    } else {
                for ($i = 0; $i < $n; $i++) {
                    $this->w[$i] = $c;
                }
            }
        }
    }

    public function get($x, $y, $d)
    {
        $ix = (($this->sx * $y) + $x) * $this->depth + $d;
        return $this->w[$ix];
    }

    public function set($x, $y, $d, $v)
    {
        $ix = (($this->sx * $y) + $x) * $this->depth + $d;
        $this->w[$ix] = $v;
    }

    public function add($x, $y, $d, $v)
    {
        $ix = (($this->sx * $y) + $x) * $this->depth + $d;
        $this->w[$ix] += $v;
    }

    public function get_grad($x, $y, $d)
    {
        $ix = (($this->sx * $y) + $x) * $this->depth + $d;
        return $this->dw[$ix];
    }

    public function set_grad($x, $y, $d, $v)
    {
        $ix = (($this->sx * $y) + $x) * $this->depth + $d;
        $this->dw[$ix] = $v;
    }

    public function add_grad($x, $y, $d, $v)
    {
        $ix = (($this->sx * $y) + $x) * $this->depth + $d;
        $this->dw[$ix] += $v;
    }

    public function cloneAndZero()
    {
        return new Vol($this->sx, $this->sy, $this->depth, 0.0);
    }

    public function clone()
    {
        $V = new Vol($this->sx, $this->sy, $this->depth, 0.0);
        $n = count($this->w);

        for ($i = 0; $i < $n; $i++) {
            $V->w[$i] = $this->w[$i];
        }

        return $V;
    }

    public function addFrom($V)
    {
        for ($k = 0; $k < count($this->w); $k++) {
            $this->w[$k] += $V->w[$k];
        }
    }

    public function addFromScaled($V, $a)
    {
        for ($k = 0; $k < count($this->w); $k++) {
            $this->w[$k] += $a * $V->w[$k];
        }
    }

    public function setConst($a)
    {
        for ($k = 0; $k < count($this->w); $k++) {
            $this->w[$k] = $a;
        }
    }

    public static function img_to_vol($img, $convert_grayscale = false)
    {
        if (gettype($img) === 'resource') {
            $image = $img;
        }

        if (gettype($img) === 'string') {
            $pathinfo = pathinfo($img);

            if ($pathinfo['extension'] === 'png') {
                $image = imagecreatefrompng($img);
            }

            if ($pathinfo['extension'] === 'jpg') {
                $image = imagecreatefromjpeg($img);
            }

            if ($pathinfo['extension'] === 'pgm') {
                $image = Pgm::loadPGM($img);
            }
        }

        // prepare the input: get pixels and normalize them
        $pv = [];
        $H = imagesy($image);
        $W = imagesx($image);

        for ($y = 0; $y < $H; $y++) {
            for ($x = 0; $x < $W; $x++) {
                $pixelRgb = imagecolorat($image, $x, $y);
                $r = ($pixelRgb >> 16) & 0xFF;
                $g = ($pixelRgb >> 8) & 0xFF;
                $b = $pixelRgb & 0xFF;

                // normalize image pixels to [-0.5, 0.5]
                $pv[] = $r / 255.0 - 0.5;
                $pv[] = $g / 255.0 - 0.5;
                $pv[] = $b / 255.0 - 0.5;
            }
        }

        $x = new Vol($W, $H, 3, 0.0); //input volume (image)
        $x->w = $pv;

        if ($convert_grayscale) {
            // flatten into depth=1 array
            $x1 = new Vol($W, $H, 1, 0.0);

            for ($i = 0; $i < $W; $i++) {
                for ($j = 0; $j < $H; $j++) {
                    $x1->set($i, $j, 0, $x->get($i, $j, 0));
                }
            }

            $x = $x1;
        }

        return $x;
    }

    public function fromJson($json)
    {
        if ($json === null) {
            return;
        }

        foreach ($json as $key => $value) {
            $this->$key = $value;
        }
    }
}
