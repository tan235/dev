MyMQL5
//+------------------------------------------------------------------+
//|                                                      ProjectName |
//|                                      Copyright 2020, CompanyName |
//|                                       http://www.companyname.net |
//+------------------------------------------------------------------+

#property copyright "Jonathan Balsicas"
#property link      "www.mannainsight.com"
#property version   "1.00"

enum Activation_Function
  {
   AFHardSigmoid = AF_HARD_SIGMOID,
   AFSigmoid = AF_SIGMOID,
   AFSwish = AF_SWISH,
   AFSoftSign = AF_SOFTSIGN,
   AFTangent = AF_TANH
  };

input group "GENERAL INPUTS"
input ENUM_TIMEFRAMES LearningTF = PERIOD_H1;
input ENUM_TIMEFRAMES PeriodTraded = PERIOD_M5;
input string MySymbol = "EURUSD";
input bool HideTesterIndicator = false;

input group "INDICATOR INPUTS"
input int RSIPeriod = 14;
input int BBPeriod = 20;
input double BBDeviation = 2.0;

input group "NN"
input int UsedBars = 1000;
input int RandomSeed = 42;
input Activation_Function ActivationFx = AFSigmoid;

int OldNumberBars, BBHandle, RSIHandle, NumOfOutputs = 2;

ulong InputNumCol = 4;

vector BBUpper, BBLower, BBMid, RSI;
vector OpenPrice, ClosePrice;
vector OutputVector, Probability;
vector LiveData(InputNumCol);

matrix InputMatrix, Weights, Bias;

double MyPoint;

bool BPDone = false;
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int OnInit()
  {
   ChartSetInteger(0,CHART_SHOW_GRID,false);
   ChartSetInteger(0,CHART_MODE,CHART_CANDLES);
   ChartSetInteger(0,CHART_COLOR_BACKGROUND,clrBlack);
   ChartSetInteger(0,CHART_COLOR_FOREGROUND,clrWhite);
   ChartSetInteger(0,CHART_COLOR_CHART_UP,clrDodgerBlue);
   ChartSetInteger(0,CHART_COLOR_CANDLE_BULL,clrDodgerBlue);
   ChartSetInteger(0,CHART_COLOR_CHART_DOWN,clrWhite);
   ChartSetInteger(0,CHART_COLOR_CANDLE_BEAR,clrWhite);
   ChartSetInteger(0,CHART_COLOR_STOP_LEVEL,clrGold);
   ChartSetInteger(0,CHART_SHOW_VOLUMES,false);
   TesterHideIndicators(HideTesterIndicator);

   MyPoint = SymbolInfoDouble(MySymbol,SYMBOL_POINT);

   RSIHandle = iRSI(MySymbol, LearningTF, RSIPeriod,PRICE_CLOSE);
   BBHandle = iBands(MySymbol, LearningTF, RSIPeriod,0,BBDeviation,PRICE_CLOSE);

   MathSrand(RandomSeed);
   GenerateParameters();

   int BarsOnChart = TerminalInfoInteger(TERMINAL_MAXBARS);
   if(UsedBars >= BarsOnChart)
     {
      if(MQLInfoInteger(MQL_TESTER)||MQLInfoInteger(MQL_OPTIMIZATION))
         printf("UsedBars is greater than bars in testing History, djust your start dates");
      else
         printf("UsedBars is greater than terminal MaxBars, go to charts section to increase the number of chart bars");
      return INIT_FAILED;
     }

   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   Comment("");

  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnTick()
  {
   if(!NewBar())
      return;
   GetLiveData();

   Probability = LiveForwardPass(LiveData);
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool NewBar()
  {
   int CurrentNumBars = Bars(MySymbol,PeriodTraded);
   if(OldNumberBars!=CurrentNumBars)
     {
      OldNumberBars = CurrentNumBars;
      return true;
     }
   return false;
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector GetLiveData()
  {
   BBUpper.CopyIndicatorBuffer(BBHandle,1,0,1);
   BBLower.CopyIndicatorBuffer(BBHandle,2,0,1);
   BBMid.CopyIndicatorBuffer(BBHandle,0,0,1);
   RSI.CopyIndicatorBuffer(RSIHandle,0,0,1);

   LiveData[0] = BBUpper[0];
   LiveData[1] = BBLower[0];
   LiveData[2] = BBMid[0];
   LiveData[3] = RSI[0];

   return LiveData;
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CollectTrainingData()
  {
   BBUpper.CopyIndicatorBuffer(BBHandle,1,0,UsedBars);
   BBLower.CopyIndicatorBuffer(BBHandle,2,0,UsedBars);
   BBMid.CopyIndicatorBuffer(BBHandle,0,0,UsedBars);
   RSI.CopyIndicatorBuffer(RSIHandle,0,0,UsedBars);

   OpenPrice.CopyRates(MySymbol,LearningTF,COPY_RATES_OPEN,0,UsedBars);
   ClosePrice.CopyRates(MySymbol,LearningTF,COPY_RATES_CLOSE,0,UsedBars);

   ulong size = OpenPrice.Size();
   OutputVector.Resize(size);

   for(ulong i=0;i<size;i++)
     {
      if(ClosePrice[i]>OpenPrice[i])
         OutputVector[i]=1;
      else
         OutputVector[i] = -1;
     }

   InputMatrix.Resize(size,InputNumCol);

   InputMatrix.Col(BBUpper,0);
   InputMatrix.Col(BBLower,1);
   InputMatrix.Col(BBMid,2);
   InputMatrix.Col(RSI,3);

   Print("Before Normalization = \n", InputMatrix);
   MinMaxNormalization(InputMatrix);
   Print("After Normalization = \n", InputMatrix);
  }
//+------------------------------------------------------------------+

struct NormalizationStructure
  {
   vector            min;
   vector            max;
   NormalizationStructure :: NormalizationStructure(ulong Columns)
     {
      min.Resize(Columns);
      max.Resize(Columns);
     }
  } MinMaxNorm(InputNumCol);

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void MinMaxNormalization(matrix &Matrix)
  {
   vector column;
   for(ulong i=0; i<InputNumCol; i++)
     {
      column = Matrix.Col(i);

      MinMaxNorm.min[i] = column.Min();
      MinMaxNorm.max[i] = column.Max();

      column = (column - MinMaxNorm.min[i])/(MinMaxNorm.max[i] - MinMaxNorm.min[i]);

      Matrix.Col(column,i);
     }
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void BPMinMaxNormalization(vector &v)
  {
   if(v.Size()!=InputNumCol)
     {
      Print("Can't Normalize the data, Vector size must be equal the size of the matrix columns");
      return;
     }
   for(ulong i=0;i<InputNumCol;i++)
      v[i] = (v[i] - MinMaxNorm.min[i])/(MinMaxNorm.max[i] - MinMaxNorm.min[i]);
  }

#define Random(mini,maxi) mini + double((MathRand()/32767.0) * (maxi-mini));

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void GenerateParameters()
  {
   Weights.Resize(NumOfOutputs, InputNumCol);
   Bias.Resize(NumOfOutputs, 1);

   for(ulong i=0; i<Weights.Rows();i++)
      for(ulong j=0; j<Weights.Cols();j++)
         Weights[i][j] = Random(-1,1);

   for(ulong i=0; i<Bias.Rows();i++)
      for(ulong j=0; j<Bias.Cols();j++)
         Bias[i][j] = Random(-1,1);

   Print("Weights = \n", Weights, "\nBias = \n", Bias);
  }
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix VectorToMatrix(const vector &v, ulong cols = 1)
  {
   ulong rows = 0;
   matrix mat = {};

   if(v.Size()%cols>0)
     {
      printf(__FUNCTION__, "Invalid Number of Columns for new matrix");
      return mat;
     }

   rows = v.Size()/cols;

   mat.Resize(rows,cols);

   for(ulong i=0, index = 0; i<rows; i++)
      for(ulong j=0, index=0; j<cols; j++)
         mat[i][j] = v[index];

   return mat;
  }
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector MatrixToVector(const matrix &Mat)
  {
   vector v = {};

   if(!v.Assign(Mat))
      Print(__FUNCTION__,"Failed converting matrix to vector");
   return v;
  }
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix Neuron(matrix &I, matrix &W, matrix&B, ENUM_ACTIVATION_FUNCTION Activation)
  {
   matrix OutPuts;
   ((W.MatMul(I)) + B).Activation(OutPuts,Activation);
   return OutPuts;
  }
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix SingleInputForwardPass(vector &input_v)
  {
   vector temp_x = input_v;

   if(!BPDone)
      BPMinMaxNormalization(temp_x);

   matrix INPUT = VectorToMatrix(temp_x);
   return Neuron(INPUT,Weights,Bias,ENUM_ACTIVATION_FUNCTION(ActivationFx));

  }
//+------------------------------------------------------------------+
vector LiveForwardPass(vector &input_v)
  {
   matrix ret_mat = SingleInputForwardPass(input_v);
   return MatrixToVector(ret_mat);
  }
//+------------------------------------------------------------------+

double CheckZero(double x) {return x==0?1.0:x;}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int SearchPattern(vector &SearchVectorA, int ValueSearchedA, vector &SearchedVectorB, int ValueSearchedB)
  {
   int count = 0;
   for(ulong i=0;i<SearchVectorA.Size();i++)
     {
      if(SearchVectorA[i]==ValueSearchedA && SearchedVectorB[i]==ValueSearchedB)
         count++;
     }
   return count;
  }
//+------------------------------------------------------------------+


//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void VectorRemoveIndex(vector &v, ulong index)
  {
   vector new_v(v.Size()-1);
   for(ulong i=0,count=0; i<v.Size();i++)
     {
      if(i!=index)
        {
         new_v[count] = v[i];
         count++;
        }
     }
   v.Copy(new_v);
  }
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void MatrixRemoveCol(matrix &mat, ulong col)
  {
   matrix new_matrix(mat.Rows(), mat.Cols()-1);
   for(ulong i=0, new_col=0;i<mat.Cols();i++)
     {
      if(i==col)
         continue;
      else
        {
         new_matrix.Col(mat.Col(i), new_col);
         new_col++;
        }
     }
   mat.Copy(new_matrix);
  }
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void MatrixRemoveRow(matrix &mat, ulong row)
  {
   matrix new_matrix(mat.Rows()-1, mat.Cols());
   for(ulong i=0, new_rows=0;i<mat.Cols();i++)
     {
      if(i==row)
         continue;
      else
        {
         new_matrix.Row(mat.Row(i), new_rows);
         new_rows++;

        }
     }
   mat.Copy(new_matrix);
  }
//+------------------------------------------------------------------+

struct ConfusionMatrixStruct
  {
   double            accuracy;
   vector            precision;
   vector            recall;
   vector            f1_score;
   vector            specificity;
   vector            support;

   vector            avg;
   vector            w_avg;

  } CMatrixObj;

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double ConfusionMatrix(vector &ActualVal, vector &PredictedVal, vector &Classes, bool report_show = true)
  {
   ulong TruePositives = 0, TrueNegatives = 0, FalsePositives = 0, FalseNegatives = 0;
   matrix ConfMatrixVal(Classes.Size(),Classes.Size());
   ConfMatrixVal.Fill(0);
   vector row(Classes.Size()),ConfVectorVal(ulong(MathPow(Classes.Size(),2)));

   if(ActualVal.Size() != PredictedVal.Size())
     {
      Print("Actual and Predicted vectors are not of the same size");
      return 0;
     }
   for(ulong i=0;i<Classes.Size();i++)
     {
      for(ulong j=0;j<Classes.Size();j++)
        {
         ConfMatrixVal[i][j] = SearchPattern(ActualVal, (int)Classes[i], PredictedVal,(int)Classes[j]);
        }
     }
//---------METRICS

//---------ACCURACY
   vector diag = ConfMatrixVal.Diag();
   CMatrixObj.accuracy = NormalizeDouble(diag.Sum()/CheckZero(ConfMatrixVal.Sum()),3);

//---------PRECISION
   CMatrixObj.precision.Resize(Classes.Size());
   vector col_v = {};
   double MetricValue = 0;
   for(ulong i=0;i<Classes.Size();i++)
     {
      col_v = ConfMatrixVal.Col(i);
      VectorRemoveIndex(col_v,i);
      TruePositives = (ulong)diag[i];
      FalsePositives = (ulong)col_v.Sum();
      MetricValue = TruePositives/CheckZero(TruePositives+FalsePositives);
      CMatrixObj.precision[i] = NormalizeDouble(MathIsValidNumber(MetricValue)?MetricValue:0,8);
     }
//---------RECALL
   vector row_v = {};
   CMatrixObj.recall.Resize(Classes.Size());

   for(ulong i=0;i<Classes.Size();i++)
     {
      row_v = ConfMatrixVal.Row(i);
      VectorRemoveIndex(row_v,i);
      TruePositives = (ulong)diag[i];
      FalseNegatives = (ulong)row_v.Sum();
      MetricValue = TruePositives/CheckZero(TruePositives+FalseNegatives);
      CMatrixObj.recall[i] = NormalizeDouble(MathIsValidNumber(MetricValue)?MetricValue:0,8);
     }

//--------Specificity
   matrix temp_mat = {};
   ZeroMemory(col_v);
   CMatrixObj.specificity.Resize(Classes.Size());

   for(ulong i=0;i<Classes.Size();i++)
     {
      temp_mat.Copy(ConfMatrixVal);
      MatrixRemoveCol(temp_mat,i);
      MatrixRemoveRow(temp_mat,i);
      col_v = ConfMatrixVal.Col(i);
      VectorRemoveIndex(col_v,i);
      FalsePositives = (ulong) col_v.Sum();
      TrueNegatives = (ulong) temp_mat.Sum();
      MetricValue = TrueNegatives/CheckZero(TrueNegatives+FalsePositives);
      CMatrixObj.specificity[i] = NormalizeDouble(MathIsValidNumber(MetricValue)?MetricValue:0,8);
     }

//--------F1 Score
   CMatrixObj.f1_score.Resize(Classes.Size());
   for(ulong i=0;i<Classes.Size();i++)
     {
      CMatrixObj.f1_score[i] = 2*(CMatrixObj.precision[i]*CMatrixObj.recall[i])/CheckZero(CMatrixObj.precision[i]-CMatrixObj.recall[i]);
      MetricValue = CMatrixObj.f1_score[i];
      CMatrixObj.f1_score[i] = NormalizeDouble(MathIsValidNumber(MetricValue)?MetricValue:0,8);
     }

//----------Support
   CMatrixObj.support.Resize(Classes.Size());
   ZeroMemory(row_v);
   for(ulong i=0;i<Classes.Size();i++)
     {
      row_v = ConfMatrixVal.Row(i);
      CMatrixObj.support[i] = NormalizeDouble(MathIsValidNumber(row_v.Sum())?row_v.Sum():0,8);

     }
   int total_size = (int)ConfMatrixVal.Sum();

//----------Average
   CMatrixObj.avg.Resize(5);
   CMatrixObj.avg[0] = CMatrixObj.precision.Mean();
   CMatrixObj.avg[1] = CMatrixObj.recall.Mean();
   CMatrixObj.avg[2] = CMatrixObj.specificity.Mean();
   CMatrixObj.avg[3] = CMatrixObj.f1_score.Mean();
   CMatrixObj.avg[4] = total_size;

//----------wAvg
   CMatrixObj.w_avg.Resize(5);
   vector Support_proportion = CMatrixObj.support/CheckZero(total_size);
   vector c = CMatrixObj.precision*Support_proportion;
   CMatrixObj.w_avg[0] = c.Sum();
   c = CMatrixObj.recall*Support_proportion;
   CMatrixObj.w_avg[1] = c.Sum();
   c = CMatrixObj.specificity*Support_proportion;
   CMatrixObj.w_avg[2] = c.Sum();
   c = CMatrixObj.f1_score*Support_proportion;
   CMatrixObj.w_avg[3] = c.Sum();
   CMatrixObj.w_avg[5] = (int)total_size;

//---------Report

   return CMatrixObj.accuracy;
  }

//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector Classes(vector &v)
  {
   vector temp_t = v,v_classes = {v[0]};
   for(ulong i=0, count=1;i<v.Size();i++)
     {
      for(ulong j=0;j<v.Size();j++)
        {
         if(v[i] == temp_t[j] && temp_t[j] != -1000)
           {
            bool count_ready = false;
            for(ulong n=0;n<v_classes.Size();n++)
               if(v[i] == v_classes[n])
                  count_ready = true;
            if(!count_ready)
              {
               count++;
               v_classes.Resize(count);
               v_classes[count-1] = v[i];
               temp_t[j] = -1000;
              }
            else
               break;
           }
         else
            continue;
        }
     }
   return v_classes;
  }
//+------------------------------------------------------------------+
