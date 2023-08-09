/*
Cleaning Data in SQL
*/


Select *
From Portfolio.dbo.NashvilleHousing

---------------------------------------------------------------------------------------------------
-- Standardize Date Format

Select SaleDateConverted, CONVERT(Date,SaleDate)
From Portfolio.dbo.NashvilleHousing

Update NashvilleHousing
SET SaleDate = CONVERT(Date,SaleDate)

ALTER TABLE NashvilleHousing
Add SaleDateConverted Date;

Update NashvilleHousing
SET SaleDateConverted = CONVERT(Date,SaleDate)


---------------------------------------------------------------------------------------------------
-- Populate Property Address Data

Select *
From Portfolio.dbo.NashvilleHousing
--Where PropertyAddress is NULL
order by ParcelID


-- ISNULL finds where property address is null in a and populates
-- it with property address from b
-- First parameter includes null values being replaced
-- Second parameter provides replacement
Select a.ParcelID, a.PropertyAddress, b.ParcelID, b.PropertyAddress, ISNULL(a.PropertyAddress,b.PropertyAddress)
From Portfolio.dbo.NashvilleHousing as a
-- Join table to itself
JOIN Portfolio.dbo.NashvilleHousing as b
-- Where parcelID is the same but its not the same row
	on a.ParcelID = b.ParcelID
	-- Unique IDs not the same
	AND a.[UniqueID ] <> b.[UniqueID ]
Where a.PropertyAddress is NULL


Update a
SET PropertyAddress = ISNULL(a.PropertyAddress,b.PropertyAddress)
From Portfolio.dbo.NashvilleHousing as a
JOIN Portfolio.dbo.NashvilleHousing as b
	on a.ParcelID = b.ParcelID
	AND a.[UniqueID ] <> b.[UniqueID ]
Where a.PropertyAddress is NULL
-- After running this, the previous query will retrun blank tables
-- Null property addresses have been filled


---------------------------------------------------------------------------------------------------

-- Breaking out PropertyAddress into individual columns (Address, City, State)
-- Using Substrings

Select PropertyAddress
From Portfolio.dbo.NashvilleHousing

-- CharIndex: parameter 1 is what we're looking for, Parameter 2 is where we're looking
-- Substring: Parameter 1 is where we're looking, Parameter 2 is where we're starting, Parameter 3 is where we're stopping
-- -1 included so that it'll stop before the comma and not include it
-- +1 included in second substring so it'll start after the comma and not include it
Select
	Substring(PropertyAddress, 1, CHARINDEX(',', PropertyAddress) - 1) As Address,
	Substring(PropertyAddress, CHARINDEX(',', PropertyAddress) + 1, LEN(PropertyAddress))  As Address
From Portfolio.dbo.NashvilleHousing

ALTER TABLE NashvilleHousing
Add PropertySplitAddress NVarchar(255);
Update NashvilleHousing
SET PropertySplitAddress = Substring(PropertyAddress, 1, CHARINDEX(',', PropertyAddress) - 1)

ALTER TABLE NashvilleHousing
Add PropertySplitCity NVarchar(255);
Update NashvilleHousing
SET PropertySplitCity = Substring(PropertyAddress, CHARINDEX(',', PropertyAddress) + 1, LEN(PropertyAddress))

Select *
From Portfolio.dbo.NashvilleHousing

---------------------------------------------------------------------------------------------------
-- Breaking out OwnerAddress into individual columns (Address, City, State)
-- Using ParseName, more simple than substring

Select OwnerAddress
From Portfolio.dbo.NashvilleHousing

-- ParseName looks for "." so we must replace commas with periods
-- Replace: Parameter1 is where we're looking, Parameter 2 is what we're replacing (comma), Parameter 3 is what we're replacing with (period)
Select
PARSENAME(REPLACE(OwnerAddress,',','.'), 3),
PARSENAME(REPLACE(OwnerAddress,',','.'), 2),
PARSENAME(REPLACE(OwnerAddress,',','.'), 1)
From Portfolio.dbo.NashvilleHousing

ALTER TABLE NashvilleHousing
Add OwnerSplitAddress NVarchar(255);

ALTER TABLE NashvilleHousing
Add OwnerSplitCity NVarchar(255);

ALTER TABLE NashvilleHousing
Add OwnerSplitState NVarchar(255);

Update NashvilleHousing
SET OwnerSplitAddress = PARSENAME(REPLACE(OwnerAddress,',','.'), 3)

Update NashvilleHousing
SET OwnerSplitCity = PARSENAME(REPLACE(OwnerAddress,',','.'), 2)

Update NashvilleHousing
SET OwnerSplitState = PARSENAME(REPLACE(OwnerAddress,',','.'), 1)


---------------------------------------------------------------------------------------------------

-- Change Y and N to Yes and No in "Sold as Vacant" Field

Select DISTINCT(SoldAsVacant), Count(SoldAsVacant)
From Portfolio.dbo.NashvilleHousing
Group By SoldAsVacant
Order By 2

-- Case: When replaces 'Y" with 'Yes' and 'N' as 'No'. Else includes values that are
-- already Yes or No so they are left alone.
Select SoldAsVacant,
CASE When SoldAsVacant = 'Y' THEN 'Yes'
	 When SoldAsVacant = 'N' THEN 'No'
	 Else SoldAsVacant
	 End
From Portfolio.dbo.NashvilleHousing
--Group By SoldAsVacant
--Order By 2

-- Update Table:
Update NashvilleHousing
SET SoldAsVacant = CASE When SoldAsVacant = 'Y' THEN 'Yes'
	 When SoldAsVacant = 'N' THEN 'No'
	 Else SoldAsVacant
	 End

---------------------------------------------------------------------------------------------------
-- Removing Duplicates

WITH RowNumCTE AS(
Select *, 
	ROW_NUMBER() OVER (
	PARTITION BY ParcelID,
				 PropertyAddress,
				 SalePrice,
				 SaleDate,
				 LegalReference
				 ORDER BY
					UniqueID
					) row_num

From Portfolio.dbo.NashvilleHousing
)
DELETE
From RowNumCTE
Where row_num > 1

---------------------------------------------------------------------------------------------------
-- Deleting Unused Columns

Select *
From Portfolio.dbo.NashvilleHousing

ALTER TABLE Portfolio.dbo.NashvilleHousing
DROP COLUMN OwnerAddress, TaxDistrict, PropertyAddress

ALTER TABLE Portfolio.dbo.NashvilleHousing
DROP COLUMN SaleDate