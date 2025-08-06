# Dagoretti North Tree Canopy Height Analysis
# Using Local TIF File

import rasterio
import rasterio.plot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import geopandas as gpd
import folium
from folium import plugins
import contextily as ctx
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import from_bounds
from rasterio.mask import mask
import json
from shapely.geometry import shape, mapping
import warnings
warnings.filterwarnings('ignore')


def load_canopy_tif(tif_path):
    """Load canopy height TIF file"""
    
    try:
        with rasterio.open(tif_path) as src:
            # Read the data
            canopy_data = src.read(1)  # Read first band
            transform = src.transform
            crs = src.crs
            bounds = src.bounds
            nodata = src.nodata
            
            print(f"TIF file loaded successfully:")
            print(f"- Shape: {canopy_data.shape}")
            print(f"- CRS: {crs}")
            print(f"- Bounds: {bounds}")
            print(f"- Data range: {np.nanmin(canopy_data):.2f} to {np.nanmax(canopy_data):.2f}")
            print(f"- NoData value: {nodata}")
            
            # Handle nodata values
            if nodata is not None:
                canopy_data = np.where(canopy_data == nodata, np.nan, canopy_data)
            
            # Remove zero and negative values (typically non-tree areas)
            canopy_data = np.where(canopy_data <= 0, np.nan, canopy_data)
            
            # Remove unrealistic values (very tall trees, likely errors)
            canopy_data = np.where(canopy_data > 50, np.nan, canopy_data)
            
            return canopy_data, transform, crs, bounds
            
    except Exception as e:
        print(f"Error loading TIF file: {e}")
        return None, None, None, None

def load_dagoretti_geojson(geojson_path=None):
    """Load Dagoretti North GeoJSON boundary (optional)"""
    
    if geojson_path and geojson_path != "":
        try:
            gdf = gpd.read_file(geojson_path)
            print(f"GeoJSON boundary loaded: {len(gdf)} features")
            return gdf
        except Exception as e:
            print(f"Error loading GeoJSON: {e}")
            return None
    else:
        print("No GeoJSON boundary provided - using full TIF extent")
        return None

def raster_to_dataframe(canopy_data, transform):
    """Convert raster data to pandas DataFrame with coordinates"""
    
    height, width = canopy_data.shape
    
    # Create coordinate arrays
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    
    # Transform pixel coordinates to geographic coordinates
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    
    # Flatten arrays and create DataFrame
    df = pd.DataFrame({
        'longitude': np.array(xs).flatten(),
        'latitude': np.array(ys).flatten(),
        'canopy_height': canopy_data.flatten()
    })
    
    df = df.dropna()
    
    print(f"Converted to DataFrame: {len(df):,} pixels with tree data")
    
    return df

def setup_visualization():
    """Setup colors and visualization parameters"""
    
    colors = [
        '#FFFFE5',  # Very light yellow (0-2m: grass/shrubs)
        '#F7FCB9',  # Light yellow-green (2-5m: bushes)
        '#D9F0A3',  # Light green (5-10m: young trees)
        '#ADDD8E',  # Medium light green (10-15m: medium trees)
        '#78C679',  # Medium green (15-20m: mature trees)
        '#31A354',  # Dark green (20-25m: tall trees)
        '#006837'   # Very dark green (25m+: very tall trees)
    ]
    
    # Create custom colormap
    cmap = LinearSegmentedColormap.from_list("canopy", colors, N=256)
    
    return colors, cmap

def analyze_canopy_statistics(df, boundary_gdf=None):
    """Analyze canopy height statistics"""
    
    print("\n=== Canopy Height Statistics for Dagoretti North ===")
    
    canopy_heights = df['canopy_height']
    
    stats = {
        'count': len(df),
        'mean': canopy_heights.mean(),
        'median': canopy_heights.median(),
        'std': canopy_heights.std(),
        'min': canopy_heights.min(),
        'max': canopy_heights.max(),
        'q25': canopy_heights.quantile(0.25),
        'q75': canopy_heights.quantile(0.75)
    }
    
    print(f"Number of pixels with trees: {stats['count']:,}")
    print(f"Mean height: {stats['mean']:.2f} m")
    print(f"Median height: {stats['median']:.2f} m")
    print(f"Standard deviation: {stats['std']:.2f} m")
    print(f"Height range: {stats['min']:.2f} - {stats['max']:.2f} m")
    print(f"25th percentile: {stats['q25']:.2f} m")
    print(f"75th percentile: {stats['q75']:.2f} m")
    
    # Height categories
    height_categories = pd.cut(canopy_heights, 
                              bins=[0, 2, 5, 10, 15, 20, 25, 50], 
                              labels=['0-2m', '2-5m', '5-10m', '10-15m', '15-20m', '20-25m', '25m+'])
    
    category_counts = height_categories.value_counts().sort_index()
    print(f"\nHeight Categories:")
    for category, count in category_counts.items():
        percentage = (count / len(canopy_heights)) * 100
        print(f"  {category}: {count:,} pixels ({percentage:.1f}%)")
    
    return stats, category_counts

def create_static_maps(df, canopy_data, transform, crs, colors, cmap, boundary_gdf=None):
    """Create static matplotlib visualizations"""
    
    # Calculate extent from transform and data shape
    height, width = canopy_data.shape
    left = transform[2]
    top = transform[5]
    right = left + width * transform[0]
    bottom = top + height * transform[4]
    extent = [left, right, bottom, top]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Dagoretti North Tree Canopy Height Analysis', fontsize=16, fontweight='bold')
    
    # Main canopy height map
    ax1 = axes[0, 0]
    im1 = ax1.imshow(canopy_data, extent=extent, cmap=cmap, 
                     vmin=0, vmax=25, interpolation='nearest')
    
    # Add boundary if available
    if boundary_gdf is not None:
        boundary_gdf.plot(ax=ax1, facecolor='none', edgecolor='red', linewidth=2)
    
    ax1.set_title('Tree Canopy Height Map')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Height (m)')
    
    # Height distribution histogram
    ax2 = axes[0, 1]
    ax2.hist(df['canopy_height'], bins=30, alpha=0.7, color='forestgreen', edgecolor='black')
    ax2.set_xlabel('Canopy Height (m)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Height Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Height categories bar chart
    ax3 = axes[1, 0]
    height_categories = pd.cut(df['canopy_height'], 
                              bins=[0, 2, 5, 10, 15, 20, 25, 50], 
                              labels=['0-2m', '2-5m', '5-10m', '10-15m', '15-20m', '20-25m', '25m+'])
    category_counts = height_categories.value_counts().sort_index()
    
    bars = ax3.bar(range(len(category_counts)), category_counts.values, 
                   color=colors[:len(category_counts)], alpha=0.8, edgecolor='black')
    ax3.set_xticks(range(len(category_counts)))
    ax3.set_xticklabels(category_counts.index, rotation=45)
    ax3.set_ylabel('Number of Pixels')
    ax3.set_title('Trees by Height Category')
    ax3.grid(True, alpha=0.3)
    
    # Add percentage labels on bars
    total = category_counts.sum()
    for bar, count in zip(bars, category_counts.values):
        percentage = (count / total) * 100
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.01, 
                f'{percentage:.1f}%', ha='center', va='bottom', fontsize=9)

    # Box plot and statistics
    ax4 = axes[1, 1]
    
    box_plot = ax4.boxplot(df['canopy_height'], vert=True, patch_artist=True)
    box_plot['boxes'][0].set_facecolor('lightgreen')
    ax4.set_ylabel('Canopy Height (m)')
    ax4.set_title('Height Distribution Box Plot')
    ax4.grid(True, alpha=0.3)
    
    stats_text = f"""
    Mean: {df['canopy_height'].mean():.1f} m
    Median: {df['canopy_height'].median():.1f} m
    Std Dev: {df['canopy_height'].std():.1f} m
    Max: {df['canopy_height'].max():.1f} m
    Min: {df['canopy_height'].min():.1f} m
    
    Total Pixels: {len(df):,}
    """
    
    ax4.text(1.3, df['canopy_height'].median(), stats_text, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
             fontsize=9, verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('dagoretti_north_canopy_analysis_static.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_interactive_folium_map(df, boundary_gdf=None):
    """Create interactive Folium map"""
    
    # Calculate center point
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=15,
        tiles='OpenStreetMap'
    )
    
    # Add satellite basemap
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google Satellite',
        name='Google Satellite',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Prepare data for heatmap (sample if too many points)
    if len(df) > 10000:
        df_sample = df.sample(n=10000, random_state=42)
        print(f"Sampling {len(df_sample)} points for interactive map")
    else:
        df_sample = df
    
    # Create heatmap data
    heat_data = [[row['latitude'], row['longitude'], row['canopy_height']] 
                 for idx, row in df_sample.iterrows()]
    
    # Add heatmap layer
    plugins.HeatMap(
        heat_data,
        min_opacity=0.4,
        max_val=df['canopy_height'].max(),
        radius=15,
        blur=10,
        gradient={
            0.0: '#FFFFE5',
            0.2: '#F7FCB9', 
            0.4: '#D9F0A3',
            0.6: '#78C679',
            0.8: '#31A354',
            1.0: '#006837'
        }
    ).add_to(m)
    
    if boundary_gdf is not None:
        # Convert to GeoJSON
        boundary_geojson = json.loads(boundary_gdf.to_json())
        
        folium.GeoJson(
            boundary_geojson,
            style_function=lambda x: {
                'fillColor': 'transparent',
                'color': 'red',
                'weight': 3,
                'fillOpacity': 0
            },
            tooltip='Dagoretti North Boundary'
        ).add_to(m)
    
    folium.LayerControl().add_to(m)
    
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 220px; height: 160px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:12px; padding: 10px; border-radius: 5px;">
    <p style="margin: 0; font-weight: bold;">Tree Canopy Height (m)</p>
    <p style="margin: 2px 0; color: #666;">Dagoretti North, Nairobi</p>
    <div style="margin-top: 8px;">
        <div><span style="background: #FFFFE5; width: 15px; height: 15px; display: inline-block; margin-right: 5px; border: 1px solid #ccc;"></span>0-2m (Grass/Shrubs)</div>
        <div><span style="background: #F7FCB9; width: 15px; height: 15px; display: inline-block; margin-right: 5px; border: 1px solid #ccc;"></span>2-5m (Bushes)</div>
        <div><span style="background: #D9F0A3; width: 15px; height: 15px; display: inline-block; margin-right: 5px; border: 1px solid #ccc;"></span>5-10m (Young trees)</div>
        <div><span style="background: #78C679; width: 15px; height: 15px; display: inline-block; margin-right: 5px; border: 1px solid #ccc;"></span>10-20m (Mature trees)</div>
        <div><span style="background: #006837; width: 15px; height: 15px; display: inline-block; margin-right: 5px; border: 1px solid #ccc;"></span>20m+ (Tall trees)</div>
    </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    m.save('dagoretti_north_canopy_interactive.html')
    print("Interactive map saved as: dagoretti_north_canopy_interactive.html")
    
    return m

def main(tif_path, geojson_path=None):
    """Main analysis function"""
    
    print("=== Dagoretti North Tree Canopy Height Analysis ===")
    print("Using Local TIF File")
    print(f"TIF file: {tif_path}")
    if geojson_path:
        print(f"Boundary file: {geojson_path}")
    
    # Load TIF data
    canopy_data, transform, crs, bounds = load_canopy_tif(tif_path)
    if canopy_data is None:
        return None
    
    # Load boundary
    boundary_gdf = load_dagoretti_geojson(geojson_path)
    
    # Convert raster to DataFrame
    df = raster_to_dataframe(canopy_data, transform)
    
    # Setup visualization
    colors, cmap = setup_visualization()
    
    # Statistical analysis
    stats, category_counts = analyze_canopy_statistics(df, boundary_gdf)
    
    print("\nCreating static visualizations...")
    fig = create_static_maps(df, canopy_data, transform, crs, colors, cmap, boundary_gdf)

    print("Creating interactive map...")
    folium_map = create_interactive_folium_map(df, boundary_gdf)
    
    # Save summary to CSV
    df.to_csv('dagoretti_north_canopy_data.csv', index=False)
    print("Data saved as: dagoretti_north_canopy_data.csv")
    
    print(f"\n=== Analysis Complete ===")
    print(f"- Static plots: dagoretti_north_canopy_analysis_static.png")
    print(f"- Interactive map: dagoretti_north_canopy_interactive.html") 
    print(f"- Raw data: dagoretti_north_canopy_data.csv")
    print(f"- Total tree pixels analyzed: {len(df):,}")
    
    return df, stats, category_counts, folium_map


if __name__ == "__main__":
    TIF_PATH = "canopy_height_dagoretti.tif"  # Replace with your TIF file path
    GEOJSON_PATH = "dagoreti.geojson"   # Replace with your GeoJSON path (optional)
    
    try:
        # Run analysis
        df, stats, categories, folium_map = main(TIF_PATH, GEOJSON_PATH)
        
    except FileNotFoundError as e:
        print(f"\nFile not found: {e}")
        print("Update the file paths in the script to match your actual files.")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Check your file paths and ensure files are valid.")