Select *
From Portfolio..CovidDeaths
Where continent is not null
ORDER BY 3,4

--Select *
--From Portfolio..CovidVaccinations
--ORDER BY 3,4

-- Select Data that we are going to use
Select Location, Date, Total_cases, New_cases, total_deaths, Population
From Portfolio..CovidDeaths
Where continent is not null
ORDER By 1,2


-- looking at total cases vs Total Deaths
-- Shows the likelihood of dying if you contract covid in your country
Select Location, Date, Total_cases, total_deaths, (total_deaths/total_cases)*100 as DeathPercentage
From Portfolio..CovidDeaths
-- Focusing on the US
Where location like '%states%' 
and continent is not null
ORDER By 1,2


-- Looking at total cases vs population
-- Showing what percentage of population got covid

Select Location, Date, Total_cases, Population, (total_cases/population)*100 as InfectedPercentage
From Portfolio..CovidDeaths
Where location like '%states%' 
and continent is not null
ORDER By 1,2


--Looking at countries with highest infection rate compared to population
Select Location, Population, MAX(Total_cases) as HighestInfectionCount, MAX((total_cases/population))*100 as InfectedPercentage
From Portfolio..CovidDeaths
WHERE Continent is not null
Group By Location, Population
ORDER By InfectedPercentage desc


--Showing countries with highest death count per population
Select Location, MAX(cast(total_deaths as bigint)) as TotalDeathCount
From Portfolio..CovidDeaths
WHERE Continent is not null
Group By Location, Population
ORDER By TotalDeathCount desc


--LET'S BREAK THINGS DOWN BY CONTINENT


-- Showing the continents with the highest death count
Select continent, MAX(cast(total_deaths as bigint)) as TotalDeathCount
From Portfolio..CovidDeaths
WHERE Continent is not null
Group By continent
ORDER By TotalDeathCount desc


-- GLOBAL NUMBERS
Select SUM(new_cases) as TotalCases, SUM(cast(new_deaths as bigint)) as TotalDeaths, SUM(cast(new_deaths as bigint))/SUM(new_cases)*100 as DeathPercentage
From Portfolio..CovidDeaths
Where continent is not null
--Group By Date
ORDER By 1,2


-- Joining tables
-- Looking at total Population Vs Vaccinations

Select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations
, SUM(CONVERT(int,vac.new_vaccinations)) OVER (Partition By dea.Location Order by dea.location,
dea.date) as RollingPeopleVaccinated
--, (RollingPeopleVaccinated/Population)*100
From Portfolio..CovidDeaths dea
Join Portfolio..CovidVaccinations vac
	On dea.location = vac.location
	and dea.date = vac.date
Where dea.continent is not null
Order by 2, 3

-- USE CTE
With PopvsVac(Continent, Location, Date, Population, New_Vaccinations, RollingPeopleVaccinated)
as
(
Select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations
, SUM(CONVERT(int,vac.new_vaccinations)) OVER (Partition By dea.Location Order by dea.location,
dea.date) as RollingPeopleVaccinated
--, (RollingPeopleVaccinated/Population)*100
From Portfolio..CovidDeaths dea
Join Portfolio..CovidVaccinations vac
	On dea.location = vac.location
	and dea.date = vac.date
Where dea.continent is not null
--Order by 2, 3
)
Select *, (RollingPeopleVaccinated/Population)*100
From PopvsVac


-- TEMP TABLE

Drop Table if exists #PercentPopulationVaccinated
Create Table #PercentPopulationVaccinated
(
Continent nvarchar(255),
Location nvarchar(255),
Date datetime,
Population numeric,
New_vacinations numeric,
RollingPeopleVaccinated numeric
)

Insert Into #PercentPopulationVaccinated
Select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations
, SUM(CONVERT(int,vac.new_vaccinations)) OVER (Partition By dea.Location Order by dea.location,
dea.date) as RollingPeopleVaccinated
--, (RollingPeopleVaccinated/Population)*100
From Portfolio..CovidDeaths dea
Join Portfolio..CovidVaccinations vac
	On dea.location = vac.location
	and dea.date = vac.date
Where dea.continent is not null
--Order by 2, 3

Select *, (RollingPeopleVaccinated/Population)*100
From #PercentPopulationVaccinated

--Creating view to store data for later visualizations
Create  View PercentPopulationVaccinatedd as
Select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations
, SUM(CONVERT(int,vac.new_vaccinations)) OVER (Partition By dea.Location Order by dea.location,
dea.date) as RollingPeopleVaccinated
--, (RollingPeopleVaccinated/Population)*100
From Portfolio..CovidDeaths dea
Join Portfolio..CovidVaccinations vac
	On dea.location = vac.location
	and dea.date = vac.date
Where dea.continent is not null
--Order by 2, 3
