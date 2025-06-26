'''
Created on Oct. 11, 2023

@author: cefect
'''
import argparse, winsound


 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='haz.rim2019.nuts3._05_nuts3_rasters')
 
    
    parser.add_argument('--basinName2', type=str,default='ems')
    parser.add_argument('--index_fp', type=str, default=None)   
    
    args = parser.parse_args()
    
    print(f"All arguments received: {args}")
    
    from haz.rim2019.nuts3._05_nuts3_rasters import run_write_rasters_perBasin as func
    
    func(args.basinName2, index_fp=args.index_fp)
    
    try:
        winsound.Beep(440, 500)
    except:
        pass