import argparse
import numpy as np

from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

from bokeh.plotting import figure, show, output_file
from bokeh.layouts import gridplot
from bokeh.palettes import Category10

# Bokeh accepts standard CSS color names or hex codes
WHEEL_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # Standard distinct palette
def parse_bag(bag_path, topic_name, target_interfaces):
    """Parses the bag and extracts timestamps and multiple joint interfaces."""
    
    # 1. Grab the base ROS 2 Jazzy typestore
    typestore = get_typestore(Stores.ROS2_JAZZY)
    
    # 2. Define the missing control_msgs definitions as strings
    interface_value_def = """
    string[] interface_names
    float64[] values
    """
    
    dynamic_joint_state_def = """
    std_msgs/Header header
    string[] joint_names
    control_msgs/InterfaceValue[] interface_values
    """
    
    # 3. Compile and register them into our typestore
    add_types = {}
    add_types.update(get_types_from_msg(interface_value_def, 'control_msgs/msg/InterfaceValue'))
    add_types.update(get_types_from_msg(dynamic_joint_state_def, 'control_msgs/msg/DynamicJointState'))
    typestore.register(add_types)
    
    timestamps = []
    parsed_data = {iface: {} for iface in target_interfaces}

    with Reader(bag_path) as reader:
        connections = [x for x in reader.connections if x.topic == topic_name]
        
        if not connections:
            raise ValueError(f"Topic '{topic_name}' not found in the bag.")

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            # 4. Now deserialize will work perfectly!
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
            
            t_sec = timestamp / 1e9
            if not timestamps:
                t0 = t_sec
            timestamps.append(t_sec - t0)

            for j_idx, joint_name in enumerate(msg.joint_names):
                interfaces = msg.interface_values[j_idx]
                
                for target_interface in target_interfaces:
                    if joint_name not in parsed_data[target_interface]:
                        parsed_data[target_interface][joint_name] = []
                        
                    try:
                        i_idx = interfaces.interface_names.index(target_interface)
                        val = interfaces.values[i_idx]
                    except ValueError:
                        val = np.nan
                        
                    parsed_data[target_interface][joint_name].append(val)

    return np.array(timestamps), parsed_data
def plot_data(timestamps, parsed_data, interfaces, output_filename="wheel_data.html"):
    """Plots the extracted wheel data using Bokeh for interactive HTML plots."""
    
    # Set the output HTML file
    output_file(output_filename, title="Robot Wheel Data")
    
    wheels = list(parsed_data[interfaces[0]].keys())[:4]
    
    plots = []
    shared_x_range = None

    for row_idx, interface in enumerate(interfaces):
        
        # Build our figure arguments dynamically
        fig_kwargs = {
            "title": f"{interface.replace('_', ' ').title()}",
            "x_axis_label": "Time (s)" if row_idx == len(interfaces) - 1 else "",
            "y_axis_label": interface,
            "height": 250,
            "width": 1200,
            "tools": "pan,box_zoom,wheel_zoom,reset,save",
            "active_drag": "box_zoom"
        }
        
        # Only pass the x_range argument if we've already created the first plot
        if shared_x_range is not None:
            fig_kwargs["x_range"] = shared_x_range
            
        # Create the figure
        p = figure(**fig_kwargs)
        
        # Capture the x_range of the first plot to share with the rest
        if shared_x_range is None:
            shared_x_range = p.x_range

        for w_idx, wheel_name in enumerate(wheels):
            if wheel_name not in parsed_data[interface]:
                continue
            
            y_data = np.array(parsed_data[interface][wheel_name])
            
            # Add the line trace
            p.line(
                timestamps, 
                y_data, 
                legend_label=wheel_name, 
                line_color=WHEEL_COLORS[w_idx % len(WHEEL_COLORS)], 
                line_width=2,
                alpha=0.8
            )
            
        # Make the legend interactive (click to hide/show trace)
        p.legend.click_policy = "hide"
        
        # If the interface data is completely empty, Bokeh will complain about 
        # putting an empty legend outside the plot, so we wrap it in a quick check
        if p.legend:
            p.add_layout(p.legend[0], 'right')
        
        # Append as a single-element list to create a 1-column grid
        plots.append([p])

    # Assemble the figures into a vertical grid layout
    grid = gridplot(plots, sizing_mode="stretch_width")
    
    # Generate the HTML and open it in the browser
    show(grid)

def main():
    parser = argparse.ArgumentParser(
        description="Extract and plot /dynamic_joint_states from a ROS 2 Jazzy bag using Bokeh.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("bag_path", type=str, help="Path to the ROS 2 bag file or directory.")
    parser.add_argument("-t", "--topic", type=str, default="/dynamic_joint_states", help="The name of the dynamic joint states topic.")
    parser.add_argument("-i", "--interfaces", type=str, nargs="+", default=['position', 'velocity', 'current', 'voltage', 'pwm'], help="Space-separated list of interface names to extract and plot.")
    parser.add_argument("-o", "--output", type=str, default="wheel_data.html", help="Name of the output HTML file.")

    args = parser.parse_args()

    print(f"Parsing bag: {args.bag_path} (Topic: {args.topic})")
    try:
        t_data, p_data = parse_bag(args.bag_path, args.topic, args.interfaces)
    except Exception as e:
        print(f"\nError parsing bag: {e}")
        return

    print(f"Generating interactive plot: {args.output}")
    plot_data(t_data, p_data, args.interfaces, args.output)

if __name__ == '__main__':
    main()
