import os
import glob
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

def input_utm_zone():
    while True:
        try:
            zone = int(input("Please enter the UTM zone (an integer between 1 and 60): "))
            if 1 <= zone <= 60:
                return zone
            else:
                print("Please enter an integer between 1 and 60.")
        except ValueError:
            print("Invalid input, please enter a number.")

def create_circle(center_x, center_y, diameter, num_points=100):
    radius = diameter / 2
    circle = Point(center_x, center_y).buffer(radius, resolution=num_points)
    return circle

def main():
    folder_path = "/storage/group/tvq5043/default/Zhuohong/NEON_mapping/MLBS_crown"  # Path to your folder containing CSV files
    merged_csv_path = "/storage/group/tvq5043/default/Zhuohong/NEON_mapping/MLBS_crown/MLBS_merged.csv"  # Path to save the merged CSV
    output_shp = "/storage/group/tvq5043/default/Zhuohong/NEON_mapping/MLBS_crown/MLBS_output_crowns.shp"  # Output shapefile path

    # Input UTM zone
    zone = input_utm_zone()
    epsg_code = 32600 + zone  # Default to northern hemisphere

    # Merge all CSV files
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    df_list = []
    for file in all_files:
        df = pd.read_csv(file)
        df_list.append(df)
    combined_df = pd.concat(df_list, ignore_index=True)

    # Save the merged CSV
    combined_df.to_csv(merged_csv_path, index=False)
    print(f"Merged CSV saved to: {merged_csv_path}")

    # Create circular geometries based on coordinates and maxCrownDiameter
    combined_df['geometry'] = combined_df.apply(
        lambda row: create_circle(row['adjEasting'], row['adjNorthing'], row['maxCrownDiameter']),
        axis=1)

    # Convert to GeoDataFrame with the specified CRS
    gdf = gpd.GeoDataFrame(combined_df, geometry='geometry', crs=f"EPSG:{epsg_code}")

    # Save the shapefile, keeping all attribute fields
    gdf.to_file(output_shp)
    print(f"Shapefile with circles saved: {output_shp}, CRS EPSG:{epsg_code}")

if __name__ == "__main__":
    main()
