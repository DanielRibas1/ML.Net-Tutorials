using Microsoft.ML;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;

namespace myApp
{
    class Program
    {
        /// <summary>
        /// Data Input for training
        /// </summary>
        public class IrisData
        {
            [Column("0")]
            public float SepalLenght;
            [Column("1")]
            public float SepalWidth;
            [Column("2")]
            public float PetalLeght;
            [Column("3")]
            public float PetalWidth;
            [Column("4")]
            [ColumnName("Label")]
            public string Label;
        }

        /// <summary>
        /// Data process result
        /// </summary>
        public class IrisPrediction
        {
            [ColumnName("PredictedLabel")]
            public string PredictedLabels;
        }

        static void Main(string[] args)
        {
            // Pipeline to load data.
            var pipeline = new LearningPipeline
            {
                new TextLoader<IrisData>("iris-data.txt", separator: ", "),

                // Transform your data
                // Assign numeric values to text in the "Label" column, because only
                // numbers can be processed during model training
                new Dictionarizer("Label"),

                // Put features in a Vector
                new ColumnConcatenator("Features",
                    nameof(IrisData.SepalLenght),
                    nameof(IrisData.SepalWidth),
                    nameof(IrisData.PetalLeght),
                    nameof(IrisData.PetalWidth)),

                // Add learning algorithm, classification scenario
                new StochasticDualCoordinateAscentClassifier(),

                // Convert Label back to original text
                new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" }
            };

            // Train model
            var model = pipeline.Train<IrisData, IrisPrediction>();

            // Predict with a done model
            var prediction = model.Predict(new IrisData()
            {
                SepalLenght = 3.3f,
                SepalWidth = 1.6f,
                PetalLeght = 0.2f,
                PetalWidth = 5.1f
            });

            Console.WriteLine($"Predicted flower is: {prediction.PredictedLabels}");
            Console.ReadLine();
        }
        
    }
}
