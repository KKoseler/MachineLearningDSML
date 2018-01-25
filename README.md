# MachineLearningDSML
## Initial code generation for an extant but untested Domain-Specific Modeling Language for Machine Learning

Specifically, a DSML defined by Breuker (2014) that is built on top of the Infer.NET C# library.
This library allows the user to define a Probabilistic Graphical Model through code and to run inference
algorithms on the model. My code generation engine is targeted towards a Baseball Analytics use case,
specifically the problem of predicting the next pitch as explored by Ganeshapillai & Guttag (2012).

These models were built using Papyrus, an open-source plugin for Eclipse that allows the user
to define a Domain-Specific Modeling Language and build models that conform to that language.
If you would like to download and examine them for yourself, ensure that you have at least version 2.0 (Neon)
of Papyrus for Eclipse. The .egx code template can be used to generate a C# file that will run inference on the
created model.
