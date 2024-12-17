# Default precision used in number conversions
import math


DEFAULT_PRECISION = 2

# Directory for writer files
writer_dir = '././././writer'

# Directory for log files
log_dir = '././././log'
# Directory for save files
save_dir = '././././save'

def number_to_value(num, units=None, precision=DEFAULT_PRECISION):
    """
    Convert a numeric value with optional units.

    Args:
        num (float): Numeric value to convert.
        units (str, optional): Unit of the value (e.g., K for thousands, M for millions).
        precision (int, optional): Number of decimal places to include. Defaults to DEFAULT_PRECISION.

    Returns:
        float: Value after conversion.
    """
    if units is None:
        if num >= 1e12:
            magnitude, units = 1e12, "T"
        elif num >= 1e9:
            magnitude, units = 1e9, "G"
        elif num >= 1e6:
            magnitude, units = 1e6, "M"
        elif num >= 1e3:
            magnitude, units = 1e3, "K"
        elif num >= 1 or num == 0:
            magnitude, units = 1, ""
        elif num >= 1e-3:
            magnitude, units = 1e-3, "m"
        else:
            magnitude, units = 1e-6, "u"
    else:
        if units == "T":
            magnitude = 1e12
        elif units == "G":
            magnitude = 1e9
        elif units == "M":
            magnitude = 1e6
        elif units == "K":
            magnitude = 1e3
        elif units == "m":
            magnitude = 1e-3
        elif units == "u":
            magnitude = 1e-6
        else:
            magnitude = 1
    return round(num / magnitude, precision)

def flops_to_value(flops, units=None, precision=DEFAULT_PRECISION):
    """
    Convert FLOPs (Floating Point Operations) with optional units.

    Args:
        flops (int): Number of FLOPs.
        units (str, optional): Unit of the value (e.g., TFLOPs, GFLOPs, MFLOPs).
        precision (int, optional): Number of decimal places to include. Defaults to DEFAULT_PRECISION.

    Returns:
        float: Value after conversion.
    """
    return (flops/1000)

def params_to_value(params_num, units=None, precision=DEFAULT_PRECISION):
    """
    Convert parameters count with optional units.

    Args:
        params_num (int): Number of parameters.
        units (str, optional): Unit of the value (e.g., KB, MB, GB).
        precision (int, optional): Number of decimal places to include. Defaults to DEFAULT_PRECISION.

    Returns:
        float: Value after conversion.
    """
    units = units.replace("B", "G") if units else units
    return (params_num)

def memory_usage_to_value(memory_bytes, units=None, precision=DEFAULT_PRECISION):
    """
    Convert memory usage with optional units.

    Args:
        memory_bytes (int): Memory usage in bytes.
        units (str, optional): Unit of the value (e.g., MB, GB).
        precision (int, optional): Number of decimal places to include. Defaults to DEFAULT_PRECISION.

    Returns:
        float: Value after conversion.
    """
    if units is None:
        units = "B"
    units = units.replace("B", "G") if units else units
    return (memory_bytes / (1024 ** 3))

def runtime_to_value(runtime_seconds, units=None, precision=DEFAULT_PRECISION):
    """
    Convert runtime with optional units.

    Args:
        runtime_seconds (float): Runtime in seconds.
        units (str, optional): Unit of the value (e.g., s, ms).
        precision (int, optional): Number of decimal places to include. Defaults to DEFAULT_PRECISION.

    Returns:
        float: Value after conversion.
    """
    if units is None:
        units = "s"
    units = units.replace("s", "ms") if units else units
    return (runtime_seconds)

def number_to_string(num, units=None, precision=DEFAULT_PRECISION):
    """
    Convert a numeric value to string format with optional units.

    Args:
        num (float): Numeric value to convert.
        units (str, optional): Unit of the value (e.g., K for thousands, M for millions).
        precision (int, optional): Number of decimal places to include. Defaults to DEFAULT_PRECISION.

    Returns:
        str: String representation of the value.
    """
    if units is None:
        if num >= 1e12:
            magnitude, units = 1e12, "T"
        elif num >= 1e9:
            magnitude, units = 1e9, "G"
        elif num >= 1e6:
            magnitude, units = 1e6, "M"
        elif num >= 1e3:
            magnitude, units = 1e3, "K"
        elif num >= 1 or num == 0:
            magnitude, units = 1, ""
        elif num >= 1e-3:
            magnitude, units = 1e-3, "m"
        else:
            magnitude, units = 1e-6, "u"
    else:
        if units == "T":
            magnitude = 1e12
        elif units == "G":
            magnitude = 1e9
        elif units == "M":
            magnitude = 1e6
        elif units == "K":
            magnitude = 1e3
        elif units == "m":
            magnitude = 1e-3
        elif units == "u":
            magnitude = 1e-6
        else:
            magnitude = 1
    return f"{round(num / magnitude, precision):g} {units}"

def convert_number(num, units=None, precision=DEFAULT_PRECISION):
    """
    Convert a numeric value to string format with optional units.

    Args:
        num (float): Numeric value to convert.
        units (str, optional): Unit of the value (e.g., K for thousands, M for millions).
        precision (int, optional): Number of decimal places to include. Defaults to DEFAULT_PRECISION.

    Returns:
        str: value with unit 
    """
    if units is None:
        magnitude, units = 1e9, "G"
        
    return round(num / magnitude, precision), units
def round_tick_interval(tick_interval):
    """Round the tick interval to a meaningful value based on its magnitude."""
    if tick_interval == 0:
        return 0

    magnitude = 10 ** math.floor(math.log10(tick_interval))
    rounded_interval = round(tick_interval / magnitude) * magnitude

    # Ensure non-zero interval
    if rounded_interval == 0:
        rounded_interval = magnitude

    return rounded_interval

def flops_to_string(flops, units=None, precision=DEFAULT_PRECISION):
    """
    Convert FLOPs (Floating Point Operations) to string format with optional units.

    Args:
        flops (int): Number of FLOPs.
        units (str, optional): Unit of the value (e.g., TFLOPs, GFLOPs, MFLOPs).
        precision (int, optional): Number of decimal places to include. Defaults to DEFAULT_PRECISION.

    Returns:
        str: String representation of the FLOPs.
    """
    return f"{number_to_string(flops, units=units, precision=precision)}FLOPS"

def params_to_string(params_num, units=None, precision=DEFAULT_PRECISION):
    """
    Convert parameters count to string format with optional units.

    Args:
        params_num (int): Number of parameters.
        units (str, optional): Unit of the value (e.g., KB, MB, GB).
        precision (int, optional): Number of decimal places to include. Defaults to DEFAULT_PRECISION.

    Returns:
        str: String representation of the parameters count.
    """
    units = units.replace("B", "G") if units else units
    return number_to_string(params_num, units=units, precision=precision).replace("G", "B").strip()

def memory_usage_to_string(memory_bytes, units=None, precision=DEFAULT_PRECISION):
    """
    Convert memory usage to string format with optional units.

    Args:
        memory_bytes (int): Memory usage in bytes.
        units (str, optional): Unit of the value (e.g., MB, GB).
        precision (int, optional): Number of decimal places to include. Defaults to DEFAULT_PRECISION.

    Returns:
        str: String representation of the memory usage.
    """
    if units is None:
        units = "B"
    units = units.replace("B", "G") if units else units
    return number_to_string(memory_bytes / (1024 ** 3), units=units).replace("G", "B").strip()

def runtime_to_string(runtime_seconds, units=None, precision=DEFAULT_PRECISION):
    """
    Convert runtime to string format with optional units.

    Args:
        runtime_seconds (float): Runtime in seconds.
        units (str, optional): Unit of the value (e.g., s, ms).
        precision (int, optional): Number of decimal places to include. Defaults to DEFAULT_PRECISION.

    Returns:
        str: String representation of the runtime.
    """
    if units is None:
        units = "s"
    units = units.replace("s", "ms") if units else units
    return number_to_string(runtime_seconds * 1000, units=units, precision=precision).replace("ms", "s").strip()


def print_model_pipline(metrics_values, units=None, precision=DEFAULT_PRECISION, print_detailed=False, model = None):
    """Prints the model metrics.

    Args:
        module_depth (int, optional): The depth of the model to which to print the aggregated module information. When set to -1, it prints information from the top to the innermost modules (the maximum depth).
        top_modules (int, optional): Limits the aggregated profile output to the number of top modules specified.
        print_detailed (bool, optional): Whether to print the detail of the model.
    """

    if 'FLOPs' in metrics_values : 
        total_flops = metrics_values['FLOPs'][-1]
        total_params = metrics_values['params'][-1]
    
    print("\n------------------------------------- Calculate metrics Results -------------------------------------")

    prints = []
            
    line_fmt = '{:<70}  {:<8}'

    if 'model_size' in metrics_values : 
        prints.append(line_fmt.format('Model Size: ', params_to_string(metrics_values['model_size'][-1])))
    if 'train_acc' in metrics_values : 
        prints.append(line_fmt.format('Final training accuracy: ', params_to_string(metrics_values['train_acc'][-1])))
    if 'valid_acc' in metrics_values : 
        prints.append(line_fmt.format('Final validation accuracy: ', params_to_string(metrics_values['valid_acc'][-1])))
    if 'runtime' in metrics_values : 
        prints.append(line_fmt.format('Total runtime: ', runtime_to_string(metrics_values['runtime'][-1])))
    if 'memory_usage' in metrics_values : 
        prints.append(line_fmt.format('Average memory usage: ', memory_usage_to_string(metrics_values['memory_usage'][-1])))
    if 'latency' in metrics_values : 
        prints.append(line_fmt.format('Latency: ', params_to_string(metrics_values['latency'][-1])))
    if 'FLOPs' in metrics_values : 
        prints.append(line_fmt.format('Total Training Params: ', params_to_string(total_params)))
        prints.append(line_fmt.format('fwd FLOPs: ', flops_to_string(total_flops, units=units,
                                                        precision=precision)))
    prints.append("---------------------------------------------------------------------------------------------------")
    
    return_print = ""
    for line in prints:
        print(line)
        return_print += line + "\n"
    return return_print
