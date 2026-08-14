import ee
import geemap
import os

# ==================== Initialize Earth Engine ====================
try:
    ee.Initialize(project='ee-lizhuohong1998')
except Exception as e:
    print(f"Initialization failed: {e}")
    ee.Authenticate()
    ee.Initialize(project='ee-lizhuohong1998')

# ==================== Custom Function ====================
def download_neon_data(site, year, output_dir):
    """
    Download NEON site data for a specified year: CHM, Refl002, RGB (intersection area)

    Parameters:
    - site: str, NEON site code, e.g., 'MLBS'
    - year: int or str, year, e.g., 2022
    - output_dir: str, folder to save downloaded files
    """

    # Load NEON datasets
    refl002 = ee.ImageCollection('projects/neon-prod-earthengine/assets/HSI_REFL/002')
    chm     = ee.ImageCollection('projects/neon-prod-earthengine/assets/CHM/001')
    rgb     = ee.ImageCollection('projects/neon-prod-earthengine/assets/RGB/001')

    # Time range
    start = f"{year}-01-01"
    end   = f"{year}-12-31"

    # Filter images (.first() gets the first image)
    refl_img = refl002.filterDate(start, end).filterMetadata('NEON_SITE', 'equals', site).first()
    chm_img  = chm.filterDate(start, end).filterMetadata('NEON_SITE', 'equals', site).first()
    rgb_img  = rgb.filterDate(start, end).filterMetadata('NEON_SITE', 'equals', site).first()

    # Null check
    if not (refl_img and chm_img and rgb_img):
        print(f"⚠️ Cannot retrieve complete data for site {site} year {year}.")
        return

    print(f"✅ Start processing: {site} - {year}")

    # Create mask: area where all three images are valid
    mask = chm_img.mask().And(
        refl_img.reduce(ee.Reducer.min()).mask()
    ).And(
        rgb_img.reduce(ee.Reducer.min()).mask()
    )

    # Apply mask
    refl_masked = refl_img.updateMask(mask)
    chm_masked  = chm_img.updateMask(mask)
    rgb_masked  = rgb_img.updateMask(mask)

    # Export region
    region = chm_masked.geometry()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Download images
    # geemap.download_ee_image(
    #     chm_masked,
    #     filename=os.path.join(output_dir, f"CHM_{site}_{year}_intersection.tif"),
    #     scale=chm_img.projection().nominalScale().getInfo(),
    #     region=region,
    #     max_tile_dim=500
    # )

    geemap.download_ee_image(
        refl_masked,
        filename=os.path.join(output_dir, f"Refl002_{site}_{year}_allbands_intersection.tif"),
        scale=refl_img.projection().nominalScale().getInfo(),
        region=region,
        max_tile_dim=500
    )

    geemap.download_ee_image(
        rgb_masked,
        filename=os.path.join(output_dir, f"RGB_{site}_{year}_intersection.tif"),
        scale=rgb_img.projection().nominalScale().getInfo(),
        region=region,
        max_tile_dim=500
    )

    print(f"✅ Download completed: {site} - {year}")

# ==================== Example Call ====================
# Set output folder (modify to your own path)
site="ABBY"
year="2023"
output_folder = '/storage/group/tvq5043/default/Zhuohong/NEON_data/'+site+year

# Example: download data for MLBS site in 2022
download_neon_data(site=site, year=year, output_dir=output_folder)

# To batch download multiple sites or years:
# for site in ['MLBS', 'SERC', 'OSBS']:
#     for year in [2022, 2023]:
#         download_neon_data(site=site, year=year, output_dir=output_folder)
