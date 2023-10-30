import os, sys
import argparse
import ROOT
from ROOT import RDataFrame

import numpy as np

# Enable multi-threading
ROOT.ROOT.EnableImplicitMT()

# Batch mode
ROOT.gROOT.SetBatch(ROOT.kTRUE)

import CMSStyle
CMSStyle.setTDRStyle() # set once

#add here your variables and desired binning
binning_dict = {
   "Ptmu3" : [0, 20, '#mu_{3} p_{T} [GeV]'],
   "pt" : [0, 20, 'p_{T} [GeV]'],
   "eta" : [-2.4, 2.4, '#eta'],
   "abs(eta)" : [0.0, 2.4, '#eta']
}

## # WIP (not efficient, will loop on whole df for each variable)
## def auto_binning(df, varname):
##   h = df1.Histo1D((varname, varname), varname)
##   h.GetQuantile
##   df.Mean(varname).GetValue()
##   df.Max(varname).GetValue()
##   df.Min(varname).GetValue()

def df_from_file(path, treename):
   if ".root" not in path:
      path = path+"/*.root"
   print(path)
   df = RDataFrame(treename, path)
   return df

def call_th1(df1, df2, sel1, sel2, f1, f2, w=''):
   print("plotting features ",f1," vs ",f2)
   #binning from dictionary
   x1_1 = binning_dict[f1][0]
   x2_1 = binning_dict[f1][1]
   x1_2 = binning_dict[f2][0]
   x2_2 = binning_dict[f2][1]
   x1 = min(x1_1, x1_2)
   x2 = max(x2_1, x2_2)

   varname1=f1.replace('(','_').replace(')','_')
   varname2=f2.replace('(','_').replace(')','_')
   #check if varname is a branch or needs definition
   if f1 not in df1.GetColumnNames():
      df1 = df1.Define(varname1, f1)
   if f2 not in df2.GetColumnNames():
      df2 = df2.Define(varname2, f2)

   h1 = df1.Filter(sel1).Histo1D((f1, f2, 50, x1, x2), varname1)
   h2 = df2.Filter(sel2).Histo1D((f2, f2, 50, x1, x2), varname2)

   return h1, h2

def draw_th1(h1, h2, f1):
   h1.Scale(1.0/h1.Integral())
   h2.Scale(1.0/h2.Integral())

   #make up missing here
   h1.SetLineColor(2)

   #from dictionary, going to use first feature for labelling
   h1.GetXaxis().SetTitle(binning_dict[f1][2])
   h1.GetYaxis().SetTitle('A.U.')

   ROOT.TGaxis.SetMaxDigits(2)

   h1.Draw("hist")
   h2.Draw("hist same")

if __name__ == "__main__":

   # Construct the argument parser
   ap = argparse.ArgumentParser()

   ap.add_argument('--file1', default='',
                   help='Name or abs. path of rootfile containing TTree')
   ap.add_argument('--treename1', default='Events',
                   help='Name of TTree')
   ap.add_argument('--varname1', default='',
                   help='Name of branch as in TTree')
   ap.add_argument('--selection1', default='',
                   help='String for selection defining first histogram')
   ap.add_argument('--label1', default='',
                   help='Label for plots (e.g. "data")')
   ap.add_argument('--file2', default='',
                   help='Name or abs. path of rootfile containing TTree')
   ap.add_argument('--treename2', default='',
                   help='Name of TTree')
   ap.add_argument('--varname2', default='',
                   help='Name of branch as in TTree')
   ap.add_argument('--selection2', default='',
                   help='String for selection defining second histogram')
   ap.add_argument('--label2', default='',
                   help='Label for plots (e.g. "MC")')

   args = ap.parse_args()

   # if file2 and treename2 are not given, we assume are same as file1
   if not args.file2:
      args.file2 = args.file1
   if not args.treename2:
      args.treename2 = args.treename1
   # if varname2 is not given, we assume is the same as varname1
   if not args.varname2:
      args.varname2 = args.varname1

   print('plotting ',args.varname1,' vs ',args.varname2)
   df1 = df_from_file(args.file1, args.treename1)
   df2 = df_from_file(args.file2, args.treename2)

   h1, h2 = call_th1(df1, df2, args.selection1, args.selection2, args.varname1, args.varname2)

   # Canvas
   canvas = ROOT.TCanvas(args.varname1, args.varname1, 800, 800)

   #if label1 and label2 are not given, we use variablename and selections
   if not args.label1:
      args.label1 = args.varname1+args.selection1
   if not args.label2:
      args.label2 = args.varname2+args.selection2

   draw_th1(h1, h2, args.varname1)

   #legend size
   dx_l = 0.45;
   dy_l = 0.3;

   #automatically place legend depending on peak position
   proba = np.array([0.0, 0.5, 1.0])
   x1 = np.array([0.0, 0.0, 0.0])
   x2 = np.array([0.0, 0.0, 0.0])
   h1.GetQuantiles(3,x1,proba)
   h2.GetQuantiles(3,x2,proba)

   range1 = h1.GetXaxis().GetXmax() - h1.GetXaxis().GetXmin()
   range2 = h2.GetXaxis().GetXmax() - h2.GetXaxis().GetXmin()

   bool1 = (h1.GetXaxis().GetXmax() - x1[1])/range1 > 0.5
   bool2 = (h2.GetXaxis().GetXmax() - x2[1])/range2 > 0.5

   if bool1 and bool2: #peak is left for both hists
       x1_l = 0.60;
   elif not(bool1 and bool2): #peak is right for both hists
       x1_l = 0.20;
   else: #place legend in the center
       x1_l = 0.40 - dx_l/2.0;

   y1_l = 0.57
   leg = ROOT.TLegend(x1_l,y1_l,x1_l+dx_l,y1_l+dy_l)
   print('TLegend(',x1_l,' ,',y1_l,' ,',x1_l+dx_l,' ,',y1_l+dy_l,')')
   leg.SetBorderSize(0)
   leg.SetFillColor(0)
   leg.SetFillStyle(0)
   leg.SetTextFont(42)
   leg.SetTextSize(0.035)

   leg.AddEntry(h1.GetPtr(),args.label1,"L")
   leg.AddEntry(h2.GetPtr(),args.label2,"L")
   canvas.cd()
   leg.Draw("SAME")
   canvas.Update()

   # CMS STYLE
   CMSStyle.setCMSEra(2022, extra="Preliminary")
   CMSStyle.setCMSLumiStyle(canvas,0)

   canvas.SaveAs('test.png')

