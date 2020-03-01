import argparse
import model
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--trainFilePath',
        type=str,
        required=True,
        help='GCS path to train csv file'
    )
    parser.add_argument(
        '--trainOutputPath',
        type=str,
        required=True,
        help='GCS path to train csv file'
    )
    parser.add_argument(
        '--testFilePath',
        type=str,
        required=True,
        help='GCS path to test csv file'

    )
    parser.add_argument(
        '--testOutputPath',
        type=str,
        required=True,
        help='GCS path to test csv file'

    )
    parser.add_argument(
       '--outputFilePath',
       type=str,
       required=True,
       help='GCS path to store the model predicted resultant output'
        
    )
    parameters = parser.parse_args()
    #print(parameters)
    model.trainAndEvaluateModel(parameters)