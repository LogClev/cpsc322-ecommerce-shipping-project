"""Provides the MyPyTable class for storing and manipulating tabular data."""

import copy
import csv
from tabulate import tabulate

class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names (list of str): M column names
        data (list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Parameters:
            column_names (list of str): initial M column names (None if empty)
            data (list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure."""
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            tuple: (N, M) where N is number of rows and M is number of columns
        """
        m = len(self.column_names)
        n = len(self.data)
        return n, m

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Parameters:
            col_identifier (str or int): string for a column name or int
                for a column index
            include_missing_values (bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Raises:
            ValueError: if col_identifier is invalid
        """
        col_values = []
        if isinstance(col_identifier, str):
            if col_identifier not in self.column_names:
                raise ValueError("The target column could not be found in the dataset")
            target_index = self.column_names.index(col_identifier)
        elif isinstance(col_identifier, int):
            if col_identifier < 0 or col_identifier >= len(self.column_names):
                raise ValueError("The target column could not be found in the dataset")
            target_index = col_identifier
        else:
            raise ValueError("The target column could not be found in the dataset")

        for row in self.data:
            if include_missing_values is True:
                col_values.append(row[target_index])
                continue
            if row[target_index] == "NA":
                continue
            col_values.append(row[target_index])
        return col_values

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leaves values as-is that cannot be converted to numeric.
        """
        for row in self.data:
            # pylint: disable-next=consider-using-enumerate
            for col in range(len(row)):
                try:
                    row[col] = float(row[col])
                except (ValueError, TypeError):
                    continue

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Parameters:
            row_indexes_to_drop (list of int): list of row indexes to remove from the table data.
        """
        # Indexes will shift while popping, need to start with the greatest index and descend
        for row_ind in sorted(row_indexes_to_drop, reverse = True):
            self.data.pop(row_ind)

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Parameters:
            filename (str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: returns self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Uses the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load.
        """
        self.data = []

        with open(filename, "r", encoding = "utf-8") as infile:
            table = list(csv.reader(infile))
        self.column_names = table.pop(0)
        for row in table:
            self.data.append(row)

        self.convert_to_numeric()
        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Parameters:
            filename (str): relative path for the CSV file to save the contents to.

        Notes:
            Uses the csv module.
        """
        with open(filename, "w", encoding = "utf-8") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(self.column_names)
            writer.writerows(self.data)

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Parameters:
            key_column_names (list of str): column names to use as row keys.

        Returns:
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
            The first instance of a row is not considered a duplicate.
        """
        found_keys = []
        duplicates = []
        key_col_ind = []
        for key in key_column_names:
            key_col_ind.append(self.column_names.index(key))

        curr_row_ind = 0
        for row in self.data:
            row_key = []
            for ind in key_col_ind:
                row_key.append(row[ind])

            if row_key not in found_keys:
                found_keys.append(row_key)
                curr_row_ind += 1
                continue

            duplicates.append(curr_row_ind)
            curr_row_ind += 1

        return duplicates

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA")."""
        curr_index = 0
        indexes_to_drop = []
        for row in self.data:
            missing_vals = False
            for col in row:
                if col == "NA":
                    missing_vals = True
            if missing_vals is True:
                indexes_to_drop.append(curr_index)
            curr_index += 1
        self.drop_rows(indexes_to_drop)

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
        by the column's original average.

        Parameters:
            col_name (str): name of column to fill with the original average (of the column).
        """
        col_index = self.column_names.index(col_name)
        values = self.get_column(col_name, False)
        value_avg = sum(values) / len(values)

        for row in self.data:
            if row[col_index] == "NA":
                row[col_index] = value_avg

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
            min: minimum of the column
            max: maximum of the column
            mid: mid-value (AKA mid-range) of the column
            avg: mean of the column
            median: median of the column

        Parameters:
            col_names (list of str): names of the numeric columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]

        Notes:
            Missing values in the columns to compute summary stats
            should be ignored.
            Assumes col_names only contains the names of columns with numeric data.
        """
        new_col_names = ["attribute", "min", "max", "mid", "avg", "median"]
        new_data = []
        for name in col_names:
            stats = []
            values = self.get_column(name, False)

            if len(values) == 0:
                continue

            stats.append(name)
            stats.append(min(values))
            stats.append(max(values))
            stats.append((min(values) + max(values)) / 2)
            stats.append(sum(values) / len(values))

            values.sort()
            middle_ind = len(values) // 2
            if len(values) % 2 == 1:
                stats.append(values[middle_ind])
            else:
                stats.append((values[middle_ind - 1] + values[middle_ind]) / 2)

            new_data.append(stats)

        return MyPyTable(new_col_names, new_data)

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
        with other_table based on key_column_names.

        Parameters:
            other_table (MyPyTable): the second table to join this table with.
            key_column_names (list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """

        new_col_names = self.column_names.copy()
        new_data = []
        key_col_ind1 = []
        key_col_ind2 = []

        for name in other_table.column_names:
            if name in new_col_names:
                continue
            new_col_names.append(name)

        for key in key_column_names:
            key_col_ind1.append(self.column_names.index(key))
            key_col_ind2.append(other_table.column_names.index(key))

        for row1 in self.data:
            row1_vals = row1.copy()
            key_col_vals1 = []
            for ind in key_col_ind1:
                key_col_vals1.append(row1[ind])

            for row2 in other_table.data:
                row2_vals = row2.copy()
                key_col_vals2 = []
                for ind in key_col_ind2:
                    key_col_vals2.append(row2[ind])

                if key_col_vals1 == key_col_vals2:
                    for ind in sorted(key_col_ind2, reverse = True):
                        row2_vals.pop(ind)
                    new_data.append(row1_vals + row2_vals)

        return MyPyTable(new_col_names, new_data)

    # pylint: disable-next=too-many-branches
    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
        other_table based on key_column_names.

        Parameters:
            other_table (MyPyTable): the second table to join this table with.
            key_column_names (list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pads attributes with missing values with "NA".
        """
        new_col_names = self.column_names.copy()
        new_data = []

        key_col_ind1 = []
        key_col_ind2 = []

        matched_ind_row2 = []

        for name in other_table.column_names:
            if name in new_col_names:
                continue
            new_col_names.append(name)

        for key in key_column_names:
            key_col_ind1.append(self.column_names.index(key))
            key_col_ind2.append(other_table.column_names.index(key))

        for row1 in self.data:
            matched = False
            row1_vals = row1.copy()

            key_col_vals1 = []
            for ind in key_col_ind1:
                key_col_vals1.append(row1[ind])

            for row2_ind, row2 in enumerate(other_table.data):
                row2_vals = row2.copy()

                key_col_vals2 = []
                for ind in key_col_ind2:
                    key_col_vals2.append(row2[ind])

                if key_col_vals1 == key_col_vals2:
                    matched = True
                    matched_ind_row2.append(row2_ind)

                    for ind in sorted(key_col_ind2, reverse=True):
                        row2_vals.pop(ind)

                    new_data.append(row1_vals + row2_vals)

            # Handling unjoined rows in table1
            if matched is False:
                num_missing_values = (len(other_table.column_names) - len(key_col_ind2))
                new_data.append(row1_vals + ["NA"] * num_missing_values)

        # Handling unjoined rows in table2
        for row2_ind, row2 in enumerate(other_table.data):
            if row2_ind not in matched_ind_row2:
                row2_vals = row2.copy()
                new_row = ["NA"] * len(self.column_names)

                # pylint: disable-next=consider-using-enumerate
                for i in range(len(key_col_ind2)):
                    value = row2[key_col_ind2[i]]
                    new_row[key_col_ind1[i]] = value

                for ind in sorted(key_col_ind2, reverse = True):
                    row2_vals.pop(ind)

                new_row = new_row + row2_vals
                new_data.append(new_row)

        return MyPyTable(new_col_names, new_data)
