'''
Created on Jul. 29, 2023

@author: cefect
'''


import argparse, winsound


 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='convert_basin_to_nrcs2')
    parser.add_argument('--basinName', type=str, default='ems', help='name of the basin')
 
 
    
    args = parser.parse_args()
    
    print(f"All arguments received: {args}")
    
    from haz.rim2019.nuts3._03_nuts3_event_selec import run_select_events_per_zone as func
    
    func(**vars(args))
    
    try:
        winsound.Beep(440, 500)
    except:
        pass
