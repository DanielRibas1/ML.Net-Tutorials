﻿using System;
using Microsoft.ML.Models;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;

namespace SentimentAnalysis
{
    class Program
    {        
        const string _dataPath = @"..\..\data\sentiment labelled sentences\imdb_labelled.txt";
        const string _testDataPath = @"..\..\data\sentiment labelled sentences\yelp_labelled.txt";

        static void Main(string[] args)
        {
            var model = TrainAndPredict();
            Evaluate(model);
            Console.WriteLine("Press any key to continue ...");
            Console.ReadLine();
        }

        public static PredictionModel<SentimentData, SentimentPrediction> TrainAndPredict()
        {
            var pipeline = new LearningPipeline
            {
                new TextLoader<SentimentData>(_dataPath, useHeader: false, separator: "tab"),
                new TextFeaturizer("Features", "SentimentText"),
                new FastTreeBinaryClassifier()
                {
                    NumLeaves = 5,
                    NumTrees = 5,
                    MinDocumentsInLeafs = 2
                }
            };

            var model = pipeline.Train<SentimentData, SentimentPrediction>();

            IEnumerable<SentimentData> sentiments = new[]
            {
                new SentimentData
                {
                    SentimentText = "Contoso's 11 is a wonderful experience",
                    Sentiment = 0
                },
                new SentimentData
                {
                    SentimentText = "The acting in this movie is very bad",
                    Sentiment = 0
                },
                new SentimentData
                {
                    SentimentText = "Joe versus the Volcano Coffee Company is a great film.",
                    Sentiment = 0
                }
            };

            var predictions = model.Predict(sentiments);

            Console.WriteLine();
            Console.WriteLine("Sentiment Predictions");
            Console.WriteLine("---------------------");

            var sentimentsAndPredictions = sentiments.Zip(predictions, (sentiment, prediction) => (sentiment, prediction));

            foreach (var (sentiment, prediction) in sentimentsAndPredictions)            
                Console.WriteLine($"Sentiment: {sentiment.SentimentText} | Prediction: {(prediction.Sentiment ? "Positive" : "Negative")}");            
            Console.WriteLine();

            return model;
        }

        public static void Evaluate(PredictionModel<SentimentData, SentimentPrediction> model)
        {
            var testData = new TextLoader<SentimentData>(_testDataPath, useHeader: false, separator: "tab");
            var evaulator = new BinaryClassificationEvaluator();
            var metrics = evaulator.Evaluate(model, testData);

            Console.WriteLine();
            Console.WriteLine("PredictionModel quality metrics evaluation");
            Console.WriteLine("------------------------------------------");
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"Auc: {metrics.Auc:P2}");
            Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
        }
    }
}
