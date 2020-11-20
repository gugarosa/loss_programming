# Programmatically Evolving Losses in Machine Learning

*This repository holds all the necessary code to run the very-same experiments described in the paper "Programmatically Evolving Losses in Machine Learning".*

---

## References

If you use our work to fulfill any of your needs, please cite us:

```
```

---

## Structure

 
 * `core`
   * `linker`
     * `gp.py`: Provides a customized Genetic Programming implementation that can use loss functions as terminals;
     * `node.py`: Provides a customized node structure that can use loss functions as terminals;
     * `space.py`: Provides a customized tree space that can use loss functions as terminals;
   * `losses.py`: Defines the losses functions that Genetic Programming can uses;
   * `model.py`: Defines the base Machine Learning architecture;
 * `models`
   * `cnn.py`: Defines a ResNet18 architecture;
   * `mlp.py`: Defines the Multi-Layer Perceptron;
 * `outputs`: Folder that holds the output files, such as `.pkl` and `.txt`;
 * `utils`
   * `loader.py`: Utility to load datasets and split them into training, validation and testing sets;
   * `object.py`: Wraps objects for usage in command line;
   * `target.py`: Implements the objective functions to be optimized;
   * `wrapper.py`: Wraps the optimization task into a single method.
   
---

## Package Guidelines

### Installation

Install all the pre-needed requirements using:

```Python
pip install -r requirements.txt
```

### Data configuration

If you wish to download the medical-based datasets, please contact us. Otherwise, you can use `torchvision` to load pre-implemented datasets.

---

## Usage

### Loss Function Optimization

The first step is to optimize a loss function using Genetic Programming guided by the validation set accuracy. To accomplish such a step, one needs to use the following script:

```Python
python model_optimization.py -h
```

*Note that `-h` invokes the script helper, which assists users in employing the appropriate parameters.*

### Evaluate the Optimized Loss Function

After conducting the optimization task, one needs to evaluate the created loss function using training and testing sets. Please, use the following script to accomplish such a procedure:

```Python
python model_evaluation.py -h
```

### Bash Script

Instead of invoking every script to conduct the experiments, it is also possible to use the provided shell script, as follows:

```Bash
./pipeline.sh
```

Such a script will conduct every step needed to accomplish the experimentation used throughout this paper. Furthermore, one can change any input argument that is defined in the script.

---

## Support

We know that we do our best, but it is inevitable to acknowledge that we make mistakes. If you ever need to report a bug, report a problem, talk to us, please do so! We will be available at our bests at this repository or gustavo.rosa@unesp.br.

---
