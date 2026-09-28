pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls
import "theme" as Room36Theme
import "pages" as Pages
import "dev" as Dev

ApplicationWindow {
    id: window

    property var appBridge: null
    property string startPage: "home"
    property bool useHomePreviewData: false
    property var projectModel: null
    property var activeProjectId: null
    property string activeProjectName: ""
    // These contracts await project/settings adapters; no persistence is implied.
    signal newProjectRequested()
    signal projectMenuRequested(var projectId)
    signal librarySortRequested(var projectId, string routeId, string option)
    signal libraryViewSizeRequested(var projectId, string routeId, string option)
    signal settingsRequested()
    readonly property string currentPage: normalizePage(startPage)

    visible: true
    width: 1440
    height: 900
    minimumWidth: metrics.minimumWindowWidth
    minimumHeight: metrics.minimumWindowHeight
    title: "Room 36"
    color: pageBackground(currentPage)

    Room36Theme.Theme { id: theme }
    Room36Theme.Metrics { id: metrics }

    function normalizePage(pageId) {
        if (pageId === "loading" || pageId === "auth" || pageId === "home" || pageId === "library" || pageId === "gallery") {
            return pageId
        }
        return "home"
    }

    function pageBackground(pageId) {
        if (pageId === "library") {
            return theme.libraryBackground
        }
        if (pageId === "auth") {
            return theme.authBackground
        }
        if (pageId === "gallery") {
            return theme.galleryBackground
        }
        if (pageId === "loading") {
            return theme.mainBlue
        }
        return theme.homeBackground
    }

    function showPage(pageId) {
        startPage = normalizePage(pageId)
    }

    function openProject(projectId, projectName) {
        if (projectId === null || projectId === undefined || projectId === "") {
            return false
        }
        activeProjectId = projectId
        activeProjectName = projectName || ""
        showPage("library")
        return true
    }

    Loader {
        id: pageLoader
        objectName: "pageLoader"
        anchors.fill: parent
        sourceComponent: {
            if (window.currentPage === "loading") {
                return loadingPageComponent
            }
            if (window.currentPage === "auth") {
                return authPageComponent
            }
            if (window.currentPage === "gallery") {
                return galleryComponent
            }
            if (window.currentPage === "library") {
                return libraryComponent
            }
            if (window.useHomePreviewData) {
                return homePreviewComponent
            }
            return homePageComponent
        }
    }

    Component {
        id: loadingPageComponent

        Pages.LoadingPage {
            objectName: "loadingPage"
        }
    }

    Component {
        id: authPageComponent

        Pages.AuthPage {
            objectName: "authPage"
        }
    }

    Component {
        id: homePageComponent

        Pages.HomePage {
            objectName: "homePage"
            projectModel: window.projectModel
            onProjectActivated: function(projectId, projectName) { window.openProject(projectId, projectName) }
            onNewProjectRequested: window.newProjectRequested()
            onProjectMenuRequested: function(projectId) { window.projectMenuRequested(projectId) }
        }
    }

    Component {
        id: homePreviewComponent

        Dev.ScreenPreview {
            objectName: "screenPreview"
            onProjectActivated: function(projectId, projectName) { window.openProject(projectId, projectName) }
            onNewProjectRequested: window.newProjectRequested()
            onProjectMenuRequested: function(projectId) { window.projectMenuRequested(projectId) }
        }
    }

    Component {
        id: libraryComponent

        Pages.LibraryShell {
            objectName: "libraryShell"
            projectId: window.activeProjectId
            projectName: window.activeProjectName
            onHomeRequested: window.showPage("home")
            onSettingsRequested: window.settingsRequested()
            onSortRequested: function(routeId, option) { window.librarySortRequested(projectId, routeId, option) }
            onViewSizeRequested: function(routeId, option) { window.libraryViewSizeRequested(projectId, routeId, option) }
        }
    }

    Component {
        id: galleryComponent

        Dev.ComponentGallery {
            objectName: "componentGallery"
            bridge: window.appBridge
        }
    }
}
